import json
import os


import torch

import triton_python_backend_utils as pb_utils

import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple
from cachetools import TTLCache
from cosyvoice.utils.common import fade_in_out
from fastcosyvoice.model import FastCosyVoice3Model
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.utils.common import TrtContextWrapper


ORIGINAL_VOCAB_SIZE = 151663
torch.set_num_threads(1)


class Prompt(NamedTuple):
    speech_tokens: torch.Tensor
    speech_feat: torch.Tensor
    spk_embedding: torch.Tensor


class Inputs(NamedTuple):
    request_id: str
    target_speech_tokens: NDArray
    prompt: Prompt
    finalize: bool | None = None
    token_offset: NDArray[np.int32] | None = None


class HIFTCache(NamedTuple):
    mel: torch.Tensor
    source: torch.Tensor
    speech: torch.Tensor


def load_trt(
    flow_decoder_estimator_model: str, device: torch.device
) -> torch.nn.Module:
    import tensorrt as trt

    with open(flow_decoder_estimator_model, "rb") as f:
        estimator_engine = trt.Runtime(
            trt.Logger(trt.Logger.INFO)
        ).deserialize_cuda_engine(f.read())
    trt_concurrent = False
    return TrtContextWrapper(
        estimator_engine, trt_concurrent=trt_concurrent, device=device
    )


def load_model(model_dir: str, device: torch.device) -> FastCosyVoice3Model:
    hyper_yaml_path = os.path.join(model_dir, "cosyvoice3.yaml")
    with open(hyper_yaml_path, "r") as f:
        configs = load_hyperpyyaml(f)

    model = load_model(model_dir, device)

    model = FastCosyVoice3Model(
        configs["llm"],
        configs["flow"],
        configs["hift"],
    )

    flow_pt_path = os.path.join(model_dir, "flow.pt")
    hift_pt_path = os.path.join(model_dir, "hift.pt")

    model.load(
        "",
        flow_pt_path,
        hift_pt_path,
        load_llm=False,
    )
    # TRT voodoo
    # sm_version = _get_gpu_sm_version()
    # flow_decoder_estimator_model = f"flow.decoder.estimator.fp32.{sm_version}.plan"
    # del model.flow.decoder.estimator
    # model.flow.decoder.estimator = load_trt(flow_decoder_estimator_model, device)
    return model


class TritonPythonModel:
    def initialize(self, args):
        # Parse model parameters
        parameters = json.loads(args["model_config"])["parameters"]
        model_params = {key: value["string_value"] for key, value in parameters.items()}
        model_dir = model_params["model_dir"]

        # Initialize device and vocoder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = load_model(model_dir, self.device)

        self._hift_cache: TTLCache[bytes, HIFTCache] = TTLCache(
            maxsize=1024,
            ttl=2 * 60,  # two minutes
        )
        self._token_hop_len = 25
        self._mel_cache_len = 8
        self._source_cache_len = int(self._mel_cache_len * 480)
        self._speech_window = np.hamming(2 * self.source_cache_len)

    def execute(
        self,
        requests: list["pb_utils.InferenceRequest"],
    ) -> list["pb_utils.InferenceResponse"]:
        return [self._infer(request) for request in requests]

    def _infer(
        self,
        request: "pb_utils.InferenceRequest",
    ) -> "pb_utils.InferenceResponse":
        inputs = self._get_inputs(request)
        tts_mel = self._model.flow.inference(
            token=inputs.target_speech_tokens,
            token_len=torch.tensor(
                [inputs.target_speech_tokens.shape[1]], dtype=torch.int32
            ).to(self.device),
            prompt_token=inputs.prompt.speech_tokens,
            prompt_token_len=torch.tensor(
                [inputs.prompt.speech_tokens.shape[1]], dtype=torch.int32
            ).to(self.device),
            prompt_feat=inputs.prompt.speech_feat,
            prompt_feat_len=torch.tensor(
                [inputs.prompt_.peech_feat.shape[1]], dtype=torch.int32
            ).to(self.device),
            embedding=inputs.prompt.spk_embedding,
            streaming=inputs.token_offset is not None,
            finalize=inputs.finalize,
        )

        if inputs.token_offset is not None:
            audio_hat = self._stream_step(tts_mel, inputs)
        else:
            audio_hat = self._full_run(tts_mel, inputs)

        wav_tensor = pb_utils.Tensor("waveform", audio_hat)
        return pb_utils.InferenceResponse(output_tensors=[wav_tensor])

    def _stream_step(
        self, tts_mel: torch.Tensor, inputs: Inputs
    ) -> NDArray[np.float32]:
        tts_mel = tts_mel[
            :, :, inputs.token_offset * self._model.flow.token_mel_ratio :
        ]

        cache: HIFTCache = self._hift_cache.get(inputs.request_id)

        if cache is None:
            hift_cache_source = torch.zeros(1, 1, 0)
            hift_cache_mel = None
            hift_cache_speech = None
        else:
            hift_cache_source = cache.source
            hift_cache_mel = cache.mel
            hift_cache_speech = cache.speech
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)

        tts_speech, tts_source = self._model.hift.inference(
            speech_feat=tts_mel,
            cache_source=hift_cache_source,
        )
        if hift_cache_speech is not None:
            tts_speech = fade_in_out(
                tts_speech,
                hift_cache_speech,
                self.token2wav_model.model.speech_window,
            )

        if inputs.finalize:
            del self._hift_cache[inputs.request_id]
        else:
            self._hift_cache[inputs.request_id] = HIFTCache(
                source=tts_mel[:, :, -self._mel_cache_len :],
                mel=tts_source[:, :, -self._source_cache_len :],
                speech=tts_speech[:, -self._source_cache_len :],
            )
            tts_speech = tts_speech[:, : -self.token2wav_model.model.source_cache_len]

        return tts_speech.squeeze(0).cpu().numpy()

    def _full_run(self, tts_mel: torch.Tensor, inputs: Inputs) -> NDArray[np.float32]:
        audio_hat, _ = self.token2wav_model.model.hift.inference(
            speech_feat=tts_mel,
            cache_source=torch.zeros(1, 1, 0),
        )
        return audio_hat.squeeze(0).cpu().numpy()

    def _get_inputs(self, request: "pb_utils.InferenceRequest") -> Inputs:
        request_id = request.request_id()
        target_speech_tokens = pb_utils.get_input_tensor_by_name(
            request, "target_speech_tokens"
        ).as_numpy()

        prompt_speech_tokens = pb_utils.get_input_tensor_by_name(
            request, "prompt_speech_tokens"
        ).as_numpy()
        prompt_speech_feat = pb_utils.get_input_tensor_by_name(
            request, "prompt_speech_feat"
        ).as_numpy()
        prompt_spk_embedding = pb_utils.get_input_tensor_by_name(
            request, "prompt_spk_embedding"
        ).as_numpy()
        target_speech_tokens_tensor = torch.from_numpy(target_speech_tokens).to(
            self.device
        )
        prompt_speech_tokens_tensor = torch.from_numpy(prompt_speech_tokens).to(
            self.device
        )
        prompt_speech_feat_tensor = torch.from_numpy(prompt_speech_feat).to(self.device)
        prompt_spk_embedding_tensor = torch.from_numpy(prompt_spk_embedding).to(
            self.device
        )
        prompt_speech_tokens = prompt_speech_tokens - ORIGINAL_VOCAB_SIZE

        # We set token_offset as an optional input to support streaming/offline tts. It has to be None when offline tts.
        token_offset_tensor = pb_utils.get_input_tensor_by_name(request, "token_offset")
        finalize_tensor = pb_utils.get_input_tensor_by_name(request, "finalize")

        token_offset, finalize = None, None
        if token_offset_tensor is not None and finalize_tensor is not None:
            token_offset = token_offset_tensor.as_numpy()
            finalize = finalize_tensor.as_numpy().item()

        return Inputs(
            request_id=request_id,
            target_speech_tokens=target_speech_tokens_tensor,
            token_offset=token_offset,
            finalize=finalize,
            prompt=Prompt(
                speech_tokens=prompt_speech_tokens_tensor,
                speech_feat=prompt_speech_feat_tensor,
                spk_embedding=prompt_spk_embedding_tensor,
            ),
        )
