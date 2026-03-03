import json
import time
from pathlib import Path
from typing import NamedTuple, Iterator
from enum import StrEnum, auto
import asyncio
import concurrent.futures

import numpy as np
from numpy.typing import NDArray
import torch
import torchaudio
import triton_python_backend_utils as pb_utils

from cosyvoicelib.llm import LLM
from cosyvoicelib.audio_tokenizer import forward_audio_tokenizer
from cosyvoicelib.features import extract_speech_feat
from cosyvoicelib.token2wav import forward_token2wav
from cosyvoicelib.speaker_embedding import forward_speaker_embedding

ORIGINAL_VOCAB_SIZE = 151663
torch.set_num_threads(1)


class DynamicChunkStrategy(StrEnum):
    EXPONENTIAL = auto()
    TIME_BASED = auto()


class Prompt(NamedTuple):
    speech_tokens: NDArray[np.int32]
    speech_feat: NDArray[np.float16]
    spk_embedding: NDArray[np.float16]


class CosyVoiceInputs(NamedTuple):
    request_id: str
    target_text: str
    reference_text: str
    prompt: Prompt


def read_wav_into_numpy(audio_path: Path):
    with Path(audio_path).open("rb") as f:
        f.read(44)
        raw_pcm = f.read()

    audio_np16 = np.frombuffer(
        raw_pcm,
        dtype=np.int16,
        count=len(raw_pcm) // 2,
    )
    return np.array([audio_np16]).astype(np.float32) / 32767


class TritonPythonModel:
    def initialize(self, args):
        self._logger = pb_utils.Logger
        # Parse model parameters
        self.model_config = json.loads(args["model_config"])
        parameters = self.model_config["parameters"]
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        self._logger.log_info(f"model_params:{model_params}")
        self.dynamic_chunk_strategy = DynamicChunkStrategy(
            model_params.get("dynamic_chunk_strategy", "exponential"),
        )

        self._logger.log_info(
            f"Using dynamic chunk strategy: {self.dynamic_chunk_strategy}"
        )
        llm_tokenizer_dir = model_params["llm_tokenizer_dir"]

        self._llm = LLM(llm_tokenizer_dir)

        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )

        self.token_frame_rate = 25
        self.flow_pre_lookahead_len = 3
        self.token_hop_len = 15

        self._reference_wav = read_wav_into_numpy("reference.wav")
        self._reference_text = (
            "You are a helpful assistant.<|endofprompt|>So my favorite podcast "
            "at the moment is a podcast called Ruined, where it's two best friends."
            " One loves horror movies, the other one hates horror movies. And so."
        )
        self._default_prompt: Prompt | None = None
    
        # Two thread pools so we don't run into situation where there is 
        # 20 workers working on execute_decoupled and None are available for
        # llm_gen thread which will lead to deadlock.
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=20) 
        self._llm_thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=20) 

    def execute(
        self,
        requests: list["pb_utils.InferenceRequest"],
    ) -> list["pb_utils.InferenceResponse"] | None:
        if self.decoupled:
            request = requests[0]
            self._thread_executor.submit(self._execute_decoupled, request)
            return

        return [self._execute(request) for request in requests]

    def _execute(self, request: "pb_utils.InferenceRequest") -> "pb_utils.InferenceResponse":
        inputs = self._get_inputs(request)
        input_ids = self._llm.parse_input(
            text=inputs.target_text,
            prompt_text=inputs.reference_text,
            prompt_speech_tokens=inputs.prompt.speech_tokens,
        )
        generated_ids = self._llm.infer(input_ids)
        return self._run(
            request_id=inputs.request_id,
            generated_ids=generated_ids,
            prompt=inputs.prompt,
        )

    def _execute_decoupled(self, request: "pb_utils.InferenceRequest") -> None:
        inputs = self._get_inputs(request)
        input_ids = self._llm.parse_input(
            text=inputs.target_text,
            prompt_text=inputs.reference_text,
            prompt_speech_tokens=inputs.prompt.speech_tokens,
        )
        generated_ids_iter = self._llm.stream_infer(input_ids)

        self._run_decoupled(
            response_sender=request.get_response_sender(),
            request_id=inputs.request_id,
            generated_ids_iter=generated_ids_iter,
            prompt=inputs.prompt,
        )
        return None

    def _get_inputs(self, request: "pb_utils.InferenceRequest") -> CosyVoiceInputs:
        if self._default_prompt is None:
            self._default_prompt = self._prepare_reference(self._reference_wav)

        request_id = request.request_id()

        target_text = pb_utils.get_input_tensor_by_name(
            request, "target_text"
        ).as_numpy()
        target_text = target_text[0][0].decode("utf-8")

        reference_text_tensor = pb_utils.get_input_tensor_by_name(
            request, "reference_text"
        )
        reference_text = self._reference_text
        if reference_text_tensor is not None:
            reference_text = reference_text_tensor.as_numpy[0][0].decode("utf-8")

        wav_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav")
        reference_wav = wav_tensor.as_numpy() if wav_tensor else None
        
        prompt = self._default_prompt
        if reference_wav is not None:
            prompt = self._prepare_reference(reference_wav)

        return CosyVoiceInputs(
            request_id=request_id,
            target_text=target_text,
            reference_text=reference_text,
            prompt=prompt,
        )

    def _prepare_reference(self, wav: NDArray[np.float32]) -> Prompt:
        wav_len = np.array([[wav.shape[-1]]], dtype=np.int32)

        speech_tokens = forward_audio_tokenizer(wav, wav_len)[None]

        speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=24000,
        )(torch.from_numpy(wav))
        speech_feat = extract_speech_feat(speech_resample)

        token_len = min(int(speech_feat.shape[1] / 2), speech_tokens.shape[-1])

        speech_tokens = speech_tokens[:, :token_len]
        speech_feat_fp16: NDArray[np.float16] = speech_feat[:, : 2 * token_len].astype(np.float16)

        spk_embedding = forward_speaker_embedding(wav)

        return Prompt(
            speech_tokens=speech_tokens,
            speech_feat=speech_feat_fp16,
            spk_embedding=spk_embedding,
        )

    async def _run(
        self,
        request_id: str,
        generated_ids: NDArray[np.int32],
        prompt: Prompt,
    ) -> "pb_utils.InferenceResponse":
        audio = forward_token2wav(
            request_id,
            generated_ids[None],
            prompt.speech_tokens,
            prompt.speech_feat,
            prompt.spk_embedding,
        )

        audio_tensor = pb_utils.Tensor("waveform", audio)
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor])

    def _run_decoupled(
        self,
        request_id: str,
        response_sender,
        generated_ids_iter: Iterator[NDArray[np.int32]],
        prompt: Prompt,
    ) -> None:
        semantic_token_ids_arr: list[int] = []
        llm_is_done_flag = [False]

        self._llm_thread_executor.submit(
            self._llm_gen,
            generated_ids_iter,
            semantic_token_ids_arr,
            llm_is_done_flag,
        )

        token_offset, chunk_index = 0, 0
        start_time = time.time()
        this_token_hop_len = self.token_hop_len
        chunk_index = -1

        while True:
            pending_num = len(semantic_token_ids_arr) - token_offset
            if llm_is_done_flag[0]:
                break

            if pending_num >= this_token_hop_len + self.flow_pre_lookahead_len:
                chunk_index += 1
                this_tts_speech_token = semantic_token_ids_arr[
                    : token_offset + this_token_hop_len + self.flow_pre_lookahead_len
                ]
                this_tts_speech_token_np = np.array([this_tts_speech_token], dtype=np.int32)

                sub_tts_speech = forward_token2wav(
                    request_id=request_id,
                    target_speech_tokens=this_tts_speech_token_np,
                    prompt_speech_tokens=prompt.speech_tokens,
                    prompt_speech_feat=prompt.speech_feat,
                    prompt_spk_embedding=prompt.spk_embedding,
                    token_offset=token_offset,
                    finalize=False,
                )

                audio_tensor = pb_utils.Tensor("waveform", sub_tts_speech)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[audio_tensor]
                )
                response_sender.send(inference_response)
                token_offset += this_token_hop_len
                this_token_hop_len = self._get_next_token_hop_len(
                    chunk_index=chunk_index,
                    this_buf_size=len(semantic_token_ids_arr),
                    this_token_hop_len=this_token_hop_len,
                    token_offset=token_offset,
                    start_time=start_time,
                )
            else:
                time.sleep(0.02)

        this_tts_speech_token_np = np.array([semantic_token_ids_arr], dtype=np.int32)
        sub_tts_speech = forward_token2wav(
            request_id=request_id,
            target_speech_tokens=this_tts_speech_token_np,
            prompt_speech_tokens=prompt.speech_tokens,
            prompt_speech_feat=prompt.speech_feat,
            prompt_spk_embedding=prompt.spk_embedding,
            token_offset=token_offset,
            finalize=True,
        )
        audio_tensor = pb_utils.Tensor("waveform", sub_tts_speech)
        inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
        response_sender.send(inference_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def _llm_gen(
        self,
        generated_ids_iter: Iterator[NDArray[np.int32]],
        semantic_token_ids_arr: list[str],
        llm_is_done_flag: list[bool]
    ) -> None:
        for generated_ids in generated_ids_iter:
            generated_ids = generated_ids.tolist()
            if len(generated_ids) == 0:
                break
            semantic_token_ids_arr.extend(generated_ids)
        llm_is_done_flag[0] = True

    def _get_next_token_hop_len(
        self,
        chunk_index: int,
        this_buf_size: int,
        this_token_hop_len: int,
        token_offset: int,
        start_time: float,
    ) -> int:
        match self.dynamic_chunk_strategy:
            case DynamicChunkStrategy.EXPONENTIAL:
                return self.token_frame_rate * (2**chunk_index)
            case DynamicChunkStrategy.TIME_BASED:
                cost_time = time.time() - start_time
                duration = token_offset / self.token_frame_rate
                if chunk_index > 0 and cost_time > 0:
                    avg_chunk_processing_time = cost_time / (chunk_index + 1)
                    if avg_chunk_processing_time > 0:
                        multiples = (duration - cost_time) / avg_chunk_processing_time
                        next_pending_num = this_buf_size - token_offset
                        if multiples > 4:
                            this_token_hop_len = (
                                next_pending_num // self.token_hop_len + 1
                            ) * self.token_hop_len
                        elif multiples > 2:
                            this_token_hop_len = (
                                next_pending_num // self.token_hop_len
                            ) * self.token_hop_len
                        else:
                            this_token_hop_len = self.token_hop_len
                        this_token_hop_len = max(self.token_hop_len, this_token_hop_len)
                return this_token_hop_len
            case _:
                raise ValueError(
                    f"Unknown DynamicChunkStrategy: {self.dynamic_chunk_strategy}"
                )

    def finalize(self):
        self._llm_thread_executor.shutdown()
        self._thread_executor.shutdown()
