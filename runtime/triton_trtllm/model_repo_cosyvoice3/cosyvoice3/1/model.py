import json
import os
import re
import time
import asyncio

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils

import httpx
import torchaudio
from functools import partial
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram


torch.set_num_threads(1)

# CosyVoice3 mel params: fmax=None (Nyquist), center=False
mel_spectrogram = partial(matcha_mel_spectrogram,
    n_fft=1920, num_mels=80, sampling_rate=24000,
    hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)


def parse_speech_token_string(response_text):
    """Parse speech tokens from string like '<|s_123|><|s_456|>' into list of int IDs."""
    speech_tokens = response_text.strip().split('><')
    if len(speech_tokens) > 1:
        speech_tokens = ['<' + t if not t.startswith('<') else t for t in speech_tokens]
        speech_tokens = [t + '>' if not t.endswith('>') else t for t in speech_tokens]
    speech_ids = []
    for token_str in speech_tokens:
        match = re.match(r'<\|s_(\d+)\|>', token_str)
        if match:
            speech_ids.append(int(match.group(1)))
    return speech_ids


class TritonPythonModel:
    """CosyVoice3 BLS orchestrator for Triton Inference Server.

    Orchestrates: audio_tokenizer, speaker_embedding, remote LLM (httpx),
    token2wav (flow-only), and vocoder (CausalHiFTGenerator).
    Supports both streaming (decoupled) and offline (non-decoupled) modes.
    """

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])
        parameters = self.model_config['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}

        self.device = torch.device("cuda")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)

        # Streaming config (L40S round-5 production: TTFA −150 ms vs upstream)
        # round-9 SYNC inherited these settings from working tree but they
        # weren't committed to 3a4eb64; restoring here.
        self.token_frame_rate = 25
        self.token_mel_ratio = 2
        # --- Re-optimization ladder knobs (config-driven; round-9 values are the
        # defaults so existing deploys are unchanged). The quality-gated ladder flips
        # these per step via config.pbtxt `parameters` to attribute each delta. ---
        self.flow_pre_lookahead_len = int(model_params.get("flow_pre_lookahead_len", 1))  # round-9=1, vanilla=3
        self.token_hop_len = int(model_params.get("token_hop_len", 8))                    # round-9=8, vanilla=15
        self.dynamic_chunk_strategy = model_params.get("dynamic_chunk_strategy", "exponential")  # vanilla="fixed"
        self.enable_trim = model_params.get("enable_trim", "1") == "1"                    # round-9=on
        self.prompt_feat_fp16 = model_params.get("prompt_feat_fp16", "1") == "1"          # round-9=fp16
        self.llm_seed = model_params.get("llm_seed", "")                                  # "" = no seed
        self.logger.log_info(f"CosyVoice3 BLS initialized, decoupled={self.decoupled}, "
                             f"chunk_strategy={self.dynamic_chunk_strategy}, "
                             f"token_hop_len={self.token_hop_len}, "
                             f"flow_pre_lookahead_len={self.flow_pre_lookahead_len}, "
                             f"enable_trim={self.enable_trim}, prompt_feat_fp16={self.prompt_feat_fp16}, "
                             f"llm_seed={self.llm_seed or 'none'}")

        # HTTP client for remote LLM (trtllm-serve default port: 8000)
        self.http_client = httpx.AsyncClient()
        self.api_base = model_params.get("llm_api_base", "http://localhost:8000/v1/chat/completions")

        # Speaker cache to avoid redundant audio_tokenizer/speaker_embedding calls
        self.speaker_cache = {}
        self.default_speaker_key = None
        self.speaker_name_to_cache_key = {}

        # Load pre-computed spk2info.pt if available
        spk2info_path = os.path.join(model_params.get("model_dir", ""), "spk2info.pt")
        if os.path.exists(spk2info_path):
            self.logger.log_info(f"Loading spk2info from {spk2info_path}")
            spk2info = torch.load(spk2info_path, map_location="cpu")
            for spk_name, spk_data in spk2info.items():
                cache_key = spk_data["reference_text"]
                self.speaker_cache[cache_key] = {
                    "prompt_speech_tokens_for_llm": spk_data["prompt_speech_tokens_for_llm"],
                    "prompt_speech_tokens": spk_data["prompt_speech_tokens"],
                    "prompt_speech_feat": spk_data["prompt_speech_feat"].to(self.device),
                    "prompt_spk_embedding": spk_data["prompt_spk_embedding"].to(self.device),
                }
                self.speaker_name_to_cache_key[spk_name] = cache_key
                if self.default_speaker_key is None:
                    self.default_speaker_key = cache_key
                self.logger.log_info(f"  Loaded speaker '{spk_name}' -> cache key: {cache_key[:60]}...")
            self.logger.log_info(f"Loaded {len(spk2info)} speaker(s) from spk2info.pt")
            self.logger.log_info(f"Available speaker names: {list(self.speaker_name_to_cache_key.keys())}")
        else:
            self.logger.log_info("No spk2info.pt found, speaker cache starts empty")

        # Tier-A A1: real-shape warmup. Earlier version used 25 random tokens —
        # didn't perfectly match production shape (chunk-1 = 9 tokens after
        # token_hop=8 + lookahead=1). Now we send dummy LLM requests with
        # production-realistic prompt lengths so TRT-LLM JIT-caches kernels
        # for actual production shapes. Lock-file prevents redundant warmups
        # when multiple BLS instances boot in parallel.
        warmup_lock = "/tmp/.bls_llm_warmup_done"
        if not os.path.exists(warmup_lock):
            try:
                import httpx as _httpx_sync
                # Three warmup variants covering common production patterns:
                #  - Short prompt (~25 tok)  → typical zero-shot reference
                #  - Medium prompt (~125 tok) → typical 5s cached speaker
                #  - Longer prompt (~252 tok) → typical 10s cached speaker
                # Each with realistic max_tokens=9 (one streaming chunk worth)
                # so generation completes fast — focus is on prefill/JIT, not gen.
                warmup_variants = [
                    (25, 16),    # short prompt, 16 gen tokens (=2 chunks)
                    (125, 64),   # medium prompt, 64 gen tokens (typical voice-chat)
                    (252, 128),  # long prompt, 128 gen tokens (longer utterance)
                ]
                self.logger.log_info("Tier-A A1: real-shape warmup (3 variants)...")
                t0 = time.time()
                for i, (prompt_tok, max_tok) in enumerate(warmup_variants):
                    spk_tokens = "".join(f"<|s_{(j*7 + i*131) % 6500}|>" for j in range(prompt_tok))
                    ref_text = (
                        "You are a helpful assistant.<|endofprompt|>"
                        f"Warmup pattern {i}, prompt {prompt_tok} tokens."
                    )
                    payload = {
                        "model": "trt_engines_bfloat16",
                        "messages": [
                            {"role": "user", "content": ref_text},
                            {"role": "assistant", "content": spk_tokens},
                        ],
                        "max_tokens": max_tok,
                        "temperature": 0.7,
                        "stream": False,
                    }
                    resp = _httpx_sync.post(self.api_base, json=payload, timeout=60.0)
                    resp.raise_for_status()
                with open(warmup_lock, "w") as f:
                    f.write(f"warmed at {time.time():.0f}")
                self.logger.log_info(f"LLM warmup OK (3 variants) in {time.time()-t0:.2f}s")
            except Exception as e:
                self.logger.log_warn(f"LLM warmup failed (continuing): {e}")

    def _convert_speech_tokens_to_str(self, speech_tokens):
        """Convert speech token IDs tensor/list to string like '<|s_N|>'."""
        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens = speech_tokens.cpu().numpy().flatten().tolist()
        return "".join(f"<|s_{int(tid)}|>" for tid in speech_tokens)

    def _extract_speech_feat(self, speech):
        """Extract mel spectrogram from 24kHz speech for flow prompt."""
        speech_feat = mel_spectrogram(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0).to(self.device)
        return speech_feat

    async def forward_llm_streaming(self, target_text, reference_text, prompt_speech_tokens):
        """Async generator: stream LLM tokens via httpx SSE."""
        full_text = f"{reference_text}{target_text}"
        prompt_speech_tokens_str = self._convert_speech_tokens_to_str(prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            # Tier-A A3: 750 → 200. Voice-chat workload produces ~50-150 speech
            # tokens. 200 is well above typical, gives less LLM scheduler memory
            # allocation overhead. Defensive cap stays in place.
            "max_tokens": 600,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": True,
        }
        if self.llm_seed:
            payload["seed"] = int(self.llm_seed)

        buffer = ""
        async with self.http_client.stream("POST", self.api_base, json=payload, timeout=None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    line_data = line[len("data: "):].strip()
                    if line_data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(line_data)
                        content = json_data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if content:
                            buffer += content
                            while True:
                                match = re.search(r"<\|s_(\d+)\|>", buffer)
                                if not match:
                                    break
                                token_num = int(match.group(1))
                                # final_id = token_num + ORIGINAL_VOCAB_SIZE
                                yield token_num
                                buffer = buffer[match.end():]
                    except json.JSONDecodeError:
                        continue

        # Flush remaining tokens
        while True:
            match = re.search(r"<\|s_(\d+)\|>", buffer)
            if not match:
                break
            token_num = int(match.group(1))
            #final_id = token_num + ORIGINAL_VOCAB_SIZE
            yield token_num
            buffer = buffer[match.end():]

    async def forward_llm_offline(self, target_text, reference_text, prompt_speech_tokens):
        """Non-streaming LLM call, returns all speech token IDs at once."""
        full_text = f"{reference_text}{target_text}"
        prompt_speech_tokens_str = self._convert_speech_tokens_to_str(prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            # Tier-A A3: 750 → 200. Voice-chat workload produces ~50-150 speech
            # tokens. 200 is well above typical, gives less LLM scheduler memory
            # allocation overhead. Defensive cap stays in place.
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": False,
        }
        if self.llm_seed:
            payload["seed"] = int(self.llm_seed)
        response = await self.http_client.post(self.api_base, json=payload, timeout=None)
        response.raise_for_status()
        response_json = response.json()
        generated_content = response_json['choices'][0]['message']['content']
        speech_ids = parse_speech_token_string(generated_content)
        # return [sid + ORIGINAL_VOCAB_SIZE for sid in speech_ids]
        return speech_ids

    def forward_audio_tokenizer(self, wav, wav_len):
        """BLS call to audio_tokenizer."""
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['prompt_speech_tokens'],
            inputs=[wav, wav_len]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_speech_tokens = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_speech_tokens')
        return torch.utils.dlpack.from_dlpack(prompt_speech_tokens.to_dlpack()).cpu()

    def forward_speaker_embedding(self, wav):
        """BLS call to speaker_embedding."""
        inference_request = pb_utils.InferenceRequest(
            model_name='speaker_embedding',
            requested_output_names=['prompt_spk_embedding'],
            inputs=[pb_utils.Tensor.from_dlpack("reference_wav", to_dlpack(wav))]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_spk_embedding = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_spk_embedding')
        return torch.utils.dlpack.from_dlpack(prompt_spk_embedding.to_dlpack())

    async def forward_token2wav(self, target_speech_tokens, prompt_speech_tokens,
                                prompt_speech_feat, prompt_spk_embedding,
                                request_id, token_offset=None, finalize=True,
                                priority=100):
        """Async BLS call to token2wav (flow-only). Returns mel tensor."""
        target_tokens_pb = pb_utils.Tensor.from_dlpack(
            "target_speech_tokens", to_dlpack(target_speech_tokens))
        prompt_tokens_pb = pb_utils.Tensor.from_dlpack(
            "prompt_speech_tokens", to_dlpack(prompt_speech_tokens))
        prompt_feat_pb = pb_utils.Tensor.from_dlpack(
            "prompt_speech_feat", to_dlpack(prompt_speech_feat))
        prompt_emb_pb = pb_utils.Tensor.from_dlpack(
            "prompt_spk_embedding", to_dlpack(prompt_spk_embedding))

        inputs = [target_tokens_pb, prompt_tokens_pb, prompt_feat_pb, prompt_emb_pb]

        if token_offset is not None:
            inputs.append(pb_utils.Tensor("token_offset",
                          np.array([[token_offset]], dtype=np.int32)))
            inputs.append(pb_utils.Tensor("finalize",
                          np.array([[finalize]], dtype=np.bool_)))

        inference_request = pb_utils.InferenceRequest(
            model_name='token2wav',
            requested_output_names=['mel'],
            inputs=inputs,
            request_id=request_id,
            parameters={"priority": priority},
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        mel = pb_utils.get_output_tensor_by_name(inference_response, 'mel')
        return torch.utils.dlpack.from_dlpack(mel.to_dlpack())

    async def forward_vocoder(self, mel, finalize):
        """Async BLS call to vocoder. Returns speech tensor."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [80, T] -> [1, 80, T]
        mel_pb = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel.float()))
        finalize_pb = pb_utils.Tensor("finalize",
                      np.array([[finalize]], dtype=np.bool_))

        inference_request = pb_utils.InferenceRequest(
            model_name='vocoder',
            requested_output_names=['tts_speech'],
            inputs=[mel_pb, finalize_pb],
        )

        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        speech = pb_utils.get_output_tensor_by_name(inference_response, 'tts_speech')
        return torch.utils.dlpack.from_dlpack(speech.to_dlpack()).cpu()

    def _prepare_prompt(self, request):
        """Extract reference audio, tokenize, compute speaker embedding and mel feat.

        If reference_wav is not provided, falls back to the default speaker
        from spk2info.pt (loaded at init).
        """
        # Check speaker_name first (highest priority)
        speaker_name_tensor = pb_utils.get_input_tensor_by_name(request, "speaker_name")
        if speaker_name_tensor is not None:
            speaker_name = speaker_name_tensor.as_numpy()[0][0].decode('utf-8').strip()
            if speaker_name:
                if speaker_name not in self.speaker_name_to_cache_key:
                    available = list(self.speaker_name_to_cache_key.keys())
                    raise pb_utils.TritonModelException(
                        f"Speaker '{speaker_name}' not found in spk2info.pt. "
                        f"Available speakers: {available}")
                cache_key = self.speaker_name_to_cache_key[speaker_name]
                cached = self.speaker_cache[cache_key]
                return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                        cached['prompt_speech_feat'], cached['prompt_spk_embedding'], cache_key)

        wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
        wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")

        reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text")
        reference_text = reference_text.as_numpy()[0][0].decode('utf-8') if reference_text is not None else ""
        if '<|endofprompt|>' not in reference_text:
            reference_text = ('You are a helpful assistant.'
                              '<|endofprompt|>') + reference_text

        # Check speaker cache
        if reference_text in self.speaker_cache:
            cached = self.speaker_cache[reference_text]
            return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                    cached['prompt_speech_feat'], cached['prompt_spk_embedding'], reference_text)

        # No reference audio — use default speaker from spk2info.pt
        if wav is None and self.default_speaker_key is not None:
            cached = self.speaker_cache[self.default_speaker_key]
            return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                    cached['prompt_speech_feat'], cached['prompt_spk_embedding'],
                    self.default_speaker_key)

        if wav is None:
            raise pb_utils.TritonModelException(
                "No reference_wav provided and no spk2info.pt loaded. "
                "Either send reference audio or generate spk2info.pt first.")

        # Audio tokenizer
        wav_np = wav.as_numpy()
        wav_len_val = wav_len.as_numpy()[0][0]
        prompt_speech_tokens = self.forward_audio_tokenizer(wav, wav_len)
        prompt_speech_tokens = prompt_speech_tokens.unsqueeze(0)  # [1, T]

        # Speaker embedding
        wav_tensor = torch.from_numpy(wav_np)
        wav_tensor = wav_tensor[:, :wav_len_val]
        prompt_spk_embedding = self.forward_speaker_embedding(wav_tensor)

        # Mel extraction at 24kHz with CosyVoice3 params
        prompt_speech_resample = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=24000)(wav_tensor)
        speech_feat = self._extract_speech_feat(prompt_speech_resample)

        # Keep full tokens for LLM prefill (untruncated)
        prompt_speech_tokens_for_llm = prompt_speech_tokens.clone()

        # Align prompt speech feat and tokens to 2:1 ratio (for flow model only)
        token_len = min(int(speech_feat.shape[1] / 2), prompt_speech_tokens.shape[-1])
        prompt_speech_feat = speech_feat[:, :2 * token_len].contiguous()
        if self.prompt_feat_fp16:
            prompt_speech_feat = prompt_speech_feat.half()
        prompt_speech_tokens = prompt_speech_tokens[:, :token_len].contiguous()

        # Cache
        self.speaker_cache[reference_text] = {
            'prompt_speech_tokens_for_llm': prompt_speech_tokens_for_llm,
            'prompt_speech_tokens': prompt_speech_tokens,
            'prompt_speech_feat': prompt_speech_feat,
            'prompt_spk_embedding': prompt_spk_embedding,
        }

        return prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, reference_text

    def _trim_leading_silence(self, wav, thr=0.02, win=240, preroll=120, fade_len=360):
        """Drop the LLM's leading pause/breath from the FIRST emitted chunk.

        The CosyVoice3 LLM emits silence speech-tokens before the content, so
        generated audio starts with 0.3-1.5 s of near-silence. We detect the
        onset on 10 ms RMS windows, cut everything before it (minus a short
        pre-roll), and Hann fade-in the new onset (also kills the chunk-0 click).
        Deterministic, reference-agnostic, and lowers TTFA. Returns the trimmed
        [1, T] waveform, or None if the whole chunk is still below threshold.

        Ladder knob: when enable_trim is False this is a passthrough (returns the
        waveform unchanged) so the baseline emits the model's true leading silence.
        """
        if not self.enable_trim:
            return wav
        x = wav[0]
        T = int(x.shape[0])
        if T == 0:
            return None
        n = T // win
        if n > 0:
            wv = x[:n * win].reshape(n, win)
            rms = torch.sqrt((wv * wv).mean(dim=1) + 1e-9)
            nz = torch.nonzero(rms > thr)
            if nz.numel() == 0:
                return None  # all silence — wait for the next chunk
            start = max(0, int(nz[0].item()) * win - preroll)
        else:
            start = 0
        out = wav[:, start:].clone()
        fl = min(fade_len, int(out.shape[1]))
        if fl > 0:
            fade = torch.hann_window(fl * 2, dtype=out.dtype, device=out.device)[:fl]
            out[:, :fl] = out[:, :fl] * fade
        return out

    async def _process_request_streaming(self, request):
        """Process a single request in streaming (decoupled) mode."""
        request_id = request.request_id()
        response_sender = request.get_response_sender()

        try:
            prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
                prompt_spk_embedding, reference_text = self._prepare_prompt(request)

            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')

            semantic_token_ids_arr = []
            token_offset = 0
            chunk_index = 0
            this_token_hop_len = self.token_hop_len
            speech_offset = 0
            speech_started = False  # flips once leading silence is trimmed away
            start_time = time.time()

            # Tier-3 H2: pre-allocated mel buffer. Avoid O(N²) torch.cat at every
            # chunk. 800 frames ≈ 16 s @ 50 Hz token-mel rate.
            MAX_MEL_FRAMES = 800
            accumulated_mel = torch.zeros(
                1, 80, MAX_MEL_FRAMES, dtype=torch.float32, device=self.device)
            mel_len = 0

            # Ladder A/B: replay a fixed gold token sequence if provided, else
            # stream from the LLM. Replaying holds the LLM constant so audio deltas
            # come only from Flow/HiFT/chunking changes under test.
            replay = self._get_replay_tokens(request)
            if replay is not None:
                async def _token_source():
                    for tid in replay:
                        yield tid
                token_source = _token_source()
            else:
                token_source = self.forward_llm_streaming(
                    target_text=target_text,
                    reference_text=reference_text,
                    prompt_speech_tokens=prompt_speech_tokens_for_llm,
                )

            async for generated_id in token_source:
                semantic_token_ids_arr.append(generated_id)

                while True:
                    pending_num = len(semantic_token_ids_arr) - token_offset
                    if pending_num < this_token_hop_len + self.flow_pre_lookahead_len:
                        break

                    # Prepare tokens for this chunk
                    end_idx = token_offset + this_token_hop_len + self.flow_pre_lookahead_len
                    this_tokens = torch.tensor(
                        semantic_token_ids_arr[:end_idx]
                    ).unsqueeze(0).to(torch.int32).to(self.device)

                    # Call token2wav (flow-only) -> mel_chunk
                    mel_chunk = await self.forward_token2wav(
                        this_tokens, prompt_speech_tokens,
                        prompt_speech_feat, prompt_spk_embedding,
                        request_id, token_offset=token_offset, finalize=False,
                        priority=chunk_index + 1,
                    )

                    # Accumulate mel in pre-alloc buffer (H2 — no torch.cat O(N²))
                    if mel_chunk.dim() == 2:
                        mel_chunk = mel_chunk.unsqueeze(0)
                    chunk_T = mel_chunk.shape[2]
                    if mel_len + chunk_T > MAX_MEL_FRAMES:
                        # Defensive 2× grow if buffer overflows
                        new_buf = torch.zeros(1, 80, MAX_MEL_FRAMES * 2,
                                              dtype=torch.float32, device=self.device)
                        new_buf[:, :, :mel_len] = accumulated_mel[:, :, :mel_len]
                        accumulated_mel = new_buf
                        MAX_MEL_FRAMES *= 2
                    accumulated_mel[:, :, mel_len:mel_len + chunk_T] = mel_chunk.to(torch.float32)
                    mel_len += chunk_T

                    # Call vocoder on valid slice (contiguous since slice is a view)
                    speech = await self.forward_vocoder(
                        accumulated_mel[:, :, :mel_len].contiguous(), finalize=False)

                    # Extract new speech
                    new_speech = speech[:, speech_offset:]
                    speech_offset += new_speech.shape[1]

                    if new_speech.shape[1] > 0:
                        if not speech_started:
                            # Trim the LLM's leading silence before the first
                            # emitted audio (also Hann fades the new onset).
                            new_speech = self._trim_leading_silence(new_speech)
                            if new_speech is not None and new_speech.shape[1] > 0:
                                speech_started = True
                        if new_speech is not None and new_speech.shape[1] > 0:
                            audio_tensor = pb_utils.Tensor.from_dlpack(
                                "waveform", to_dlpack(new_speech))
                            inference_response = pb_utils.InferenceResponse(
                                output_tensors=[audio_tensor])
                            response_sender.send(inference_response)

                    token_offset += this_token_hop_len

                    # Dynamic chunk strategy
                    if self.dynamic_chunk_strategy == "exponential":
                        this_token_hop_len = self.token_frame_rate * (2 ** chunk_index)
                    elif self.dynamic_chunk_strategy == "time_based":
                        cost_time = time.time() - start_time
                        duration = token_offset / self.token_frame_rate
                        if chunk_index > 0 and cost_time > 0:
                            avg_chunk_time = cost_time / (chunk_index + 1)
                            if avg_chunk_time > 0:
                                multiples = (duration - cost_time) / avg_chunk_time
                                next_pending = len(semantic_token_ids_arr) - token_offset
                                if multiples > 4:
                                    this_token_hop_len = (next_pending // self.token_hop_len + 1) * self.token_hop_len
                                elif multiples > 2:
                                    this_token_hop_len = (next_pending // self.token_hop_len) * self.token_hop_len
                                else:
                                    this_token_hop_len = self.token_hop_len
                                this_token_hop_len = max(self.token_hop_len, this_token_hop_len)

                    chunk_index += 1

            # Final chunk with remaining tokens
            if len(semantic_token_ids_arr) > 0:
                remaining_tokens = torch.tensor(
                    semantic_token_ids_arr
                ).unsqueeze(0).to(torch.int32).to(self.device)

                mel_chunk = await self.forward_token2wav(
                    remaining_tokens, prompt_speech_tokens,
                    prompt_speech_feat, prompt_spk_embedding,
                    request_id, token_offset=token_offset, finalize=True,
                    priority=chunk_index + 1,
                )

                if mel_chunk.dim() == 2:
                    mel_chunk = mel_chunk.unsqueeze(0)
                chunk_T = mel_chunk.shape[2]
                if mel_len + chunk_T > MAX_MEL_FRAMES:
                    new_buf = torch.zeros(1, 80, MAX_MEL_FRAMES * 2,
                                          dtype=torch.float32, device=self.device)
                    new_buf[:, :, :mel_len] = accumulated_mel[:, :, :mel_len]
                    accumulated_mel = new_buf
                    MAX_MEL_FRAMES *= 2
                accumulated_mel[:, :, mel_len:mel_len + chunk_T] = mel_chunk.to(torch.float32)
                mel_len += chunk_T

                speech = await self.forward_vocoder(
                    accumulated_mel[:, :, :mel_len].contiguous(), finalize=True)

                new_speech = speech[:, speech_offset:]
                if not speech_started and new_speech.shape[1] > 0:
                    # Short utterance: all audio arrived in the final chunk —
                    # trim its leading silence too.
                    new_speech = self._trim_leading_silence(new_speech)
                    if new_speech is not None and new_speech.shape[1] > 0:
                        speech_started = True
                if new_speech is not None and new_speech.shape[1] > 0:
                    audio_tensor = pb_utils.Tensor.from_dlpack(
                        "waveform", to_dlpack(new_speech))
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[audio_tensor])
                    response_sender.send(inference_response)

            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        except Exception as e:
            self.logger.log_error(f"Error in streaming request: {e}")
            error_response = pb_utils.InferenceResponse(
                error=pb_utils.TritonError(str(e)))
            response_sender.send(error_response)
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    async def _process_request_offline(self, request):
        """Process a single request in offline (non-decoupled) mode."""
        request_id = request.request_id()

        prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
            prompt_spk_embedding, reference_text = self._prepare_prompt(request)

        target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
        target_text = target_text[0][0].decode('utf-8')

        # Ladder A/B: if replay_tokens is provided, skip the LLM and render the
        # fixed gold token sequence so audio deltas are attributable to Flow/HiFT/
        # chunking only. Otherwise generate normally (and emit the tokens so the
        # gold pass can be dumped).
        replay = self._get_replay_tokens(request)
        if replay is not None:
            all_token_ids = replay
        else:
            all_token_ids = await self.forward_llm_offline(
                target_text=target_text,
                reference_text=reference_text,
                prompt_speech_tokens=prompt_speech_tokens_for_llm,
            )

        if len(all_token_ids) == 0:
            raise pb_utils.TritonModelException("LLM generated no speech tokens")

        all_tokens = torch.tensor(all_token_ids).unsqueeze(0).to(torch.int32).to(self.device)

        # token2wav (no token_offset, finalize=True) -> full mel
        mel = await self.forward_token2wav(
            all_tokens, prompt_speech_tokens,
            prompt_speech_feat, prompt_spk_embedding,
            request_id,
        )

        # vocoder -> full speech
        speech = await self.forward_vocoder(mel, finalize=True)

        # Trim the LLM's leading silence (deterministic, reference-agnostic).
        trimmed = self._trim_leading_silence(speech)
        if trimmed is not None and trimmed.shape[1] > 0:
            speech = trimmed

        audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(speech))
        # Also emit the speech tokens so the gold pass can dump them for replay A/B.
        tokens_tensor = pb_utils.Tensor(
            "speech_tokens", np.array([all_token_ids], dtype=np.int32))
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor, tokens_tensor])

    def _get_replay_tokens(self, request):
        """Return a python list of speech-token ids from the optional replay_tokens
        input, or None if not provided. Used by the ladder A/B harness to hold the
        LLM output constant while varying Flow/HiFT/chunking."""
        t = pb_utils.get_input_tensor_by_name(request, "replay_tokens")
        if t is None:
            return None
        arr = t.as_numpy().reshape(-1).astype(np.int64)
        return arr.tolist() if arr.size > 0 else None

    async def execute(self, requests):
        if self.decoupled:
            tasks = [
                asyncio.create_task(self._process_request_streaming(request))
                for request in requests
            ]
            await asyncio.gather(*tasks)
            return None
        else:
            responses = []
            for request in requests:
                try:
                    response = await self._process_request_offline(request)
                    responses.append(response)
                except Exception as e:
                    self.logger.log_error(f"Error in offline request: {e}")
                    responses.append(pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))))
            return responses

    def finalize(self):
        self.logger.log_info("Finalizing CosyVoice3 BLS model")
        if hasattr(self, "http_client"):
            asyncio.run(self.http_client.aclose())
