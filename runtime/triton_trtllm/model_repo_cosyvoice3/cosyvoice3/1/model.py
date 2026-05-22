import json
import os
import re
import time
import asyncio
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack
import triton_python_backend_utils as pb_utils

import httpx
import torchaudio
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram


torch.set_num_threads(1)

# CosyVoice3 mel params: fmax=None (Nyquist), center=False
mel_spectrogram = partial(matcha_mel_spectrogram,
    n_fft=1920, num_mels=80, sampling_rate=24000,
    hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)

# Pre-compiled regex for SSE speech-token parsing (Tier-1 H4).
# Was per-call re.search() which recompiled on every line.
_TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")


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
        self.token_frame_rate = 25
        self.flow_pre_lookahead_len = 1
        self.token_hop_len = 8
        self.token_mel_ratio = 2
        self.dynamic_chunk_strategy = model_params.get("dynamic_chunk_strategy", "exponential")

        # Tier-2 A5: bounded concurrency per BLS instance via semaphore.
        # 16 BLS × 8 concurrent inflight each = 128 max concurrent streams (cluster-wide).
        # Without this, burst N=32+ catastrophically overloads downstream models with
        # backed-up requests → p99 tail explodes. Semaphore makes overflow graceful.
        max_concurrent = int(model_params.get("bls_max_concurrent", "8"))
        self._stream_semaphore = asyncio.Semaphore(max_concurrent)
        self.logger.log_info(f"BLS bounded concurrency: max_concurrent={max_concurrent}/instance")

        # Tier-4 BLS coordinator state (real GPU batching for token2wav).
        # When multiple concurrent streams in the same BLS event loop emit
        # forward_token2wav requests, the coordinator opportunistically batches them
        # by shape into ONE pb_utils.InferenceRequest (with batched tensors).
        # token2wav.execute() unpacks the batch internally via flow.inference_batched().
        # This bypasses Triton dynamic_batching (which couldn't accumulate due to
        # inter-arrival > queue_delay) — we batch in-process opportunistically.
        self._t2w_queue = asyncio.Queue()
        self._t2w_dispatcher_task = None
        self._t2w_max_batch = int(model_params.get("t2w_max_batch", "8"))
        # Wait window after first item to let other streams accumulate.
        # 0 = pure opportunistic (no TTFA cost, batching only when another stream
        # pushes BEFORE first item is observed). 5-15ms catches near-simultaneous
        # pushes from streams in same BLS event loop. Default 0 — most workloads
        # have variable prompts so batching wouldn't engage anyway (see
        # _t2w_shape_key — inference_batched asserts uniform prompt/target lens).
        # Set higher when production traffic uses cached speakers (spk2info.pt).
        self._t2w_wait_ms = int(model_params.get("t2w_dispatch_wait_ms", "0"))
        # Same pattern for vocoder. Vocoder's shape_key is just (mel_T, finalize)
        # since vocoder takes no prompt — broader batching opportunity than t2w.
        self._voc_queue = asyncio.Queue()
        self._voc_dispatcher_task = None
        self._voc_max_batch = int(model_params.get("voc_max_batch", "8"))
        self._voc_wait_ms = int(model_params.get("voc_dispatch_wait_ms", "0"))
        # Per-instance ID for diagnostics (random 4 hex chars per process)
        import secrets
        self._instance_id = secrets.token_hex(2)
        self._t2w_push_count = 0
        self._t2w_dispatch_count = 0
        self._voc_push_count = 0
        self._voc_dispatch_count = 0
        self.logger.log_info(
            f"[inst {self._instance_id}] coord cfg: "
            f"t2w_wait={self._t2w_wait_ms}ms voc_wait={self._voc_wait_ms}ms "
            f"t2w_max_batch={self._t2w_max_batch} voc_max_batch={self._voc_max_batch}")
        self.logger.log_info(f"CosyVoice3 BLS initialized, decoupled={self.decoupled}, "
                             f"chunk_strategy={self.dynamic_chunk_strategy}")

        # HTTP client for remote LLM (Tier-1 C1):
        # - explicit Limits prevent connection-pool exhaustion under burst
        # - explicit Timeouts fail fast on stuck connections (was None → could hang forever)
        # - HTTP/2 multiplexes streams over one TCP session — lower latency + fewer FDs
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=64,
                max_connections=128,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(connect=2.0, read=60.0, write=5.0, pool=5.0),
            http2=True,
        )
        self.api_base = model_params.get("llm_api_base", "http://localhost:8000/v1/chat/completions")

        # Speaker cache with LRU eviction (Tier-1 C3) and per-key async locks (C2).
        # OrderedDict → predictable eviction, prevents unbounded GPU memory growth
        # when prod sees many distinct reference audios.
        self.speaker_cache = OrderedDict()
        self.cache_max_size = int(model_params.get("speaker_cache_max_size", "256"))
        self._cache_locks = {}   # per-key asyncio.Lock for race-free fill
        self.default_speaker_key = None
        self.speaker_name_to_cache_key = {}

        # Cached resampler (Tier-1 M4). Was created on every cold _prepare_prompt
        # call — re-compiles internal kaiser filters each time. Now built once.
        self.resampler_16to24 = torchaudio.transforms.Resample(
            orig_freq=16000, new_freq=24000)

        # Load pre-computed spk2info.pt if available
        spk2info_path = os.path.join(model_params.get("model_dir", ""), "spk2info.pt")
        if os.path.exists(spk2info_path):
            self.logger.log_info(f"Loading spk2info from {spk2info_path}")
            spk2info = torch.load(spk2info_path, map_location="cpu")
            for spk_name, spk_data in spk2info.items():
                cache_key = spk_data["reference_text"]
                entry = {
                    "prompt_speech_tokens_for_llm": spk_data["prompt_speech_tokens_for_llm"],
                    "prompt_speech_tokens": spk_data["prompt_speech_tokens"],
                    "prompt_speech_feat": spk_data["prompt_speech_feat"].to(self.device),
                    "prompt_spk_embedding": spk_data["prompt_spk_embedding"].to(self.device),
                }
                # Tier-A A1: pre-baked LLM token string. Old spk2info entries
                # without this field — fall back to runtime convert (slower).
                if "prompt_speech_tokens_str" in spk_data:
                    entry["prompt_speech_tokens_str"] = spk_data["prompt_speech_tokens_str"]
                self.speaker_cache[cache_key] = entry
                self.speaker_name_to_cache_key[spk_name] = cache_key
                if self.default_speaker_key is None:
                    self.default_speaker_key = cache_key
                self.logger.log_info(f"  Loaded speaker '{spk_name}' -> cache key: {cache_key[:60]}...")
            self.logger.log_info(f"Loaded {len(spk2info)} speaker(s) from spk2info.pt")
            self.logger.log_info(f"Available speaker names: {list(self.speaker_name_to_cache_key.keys())}")
        else:
            self.logger.log_info("No spk2info.pt found, speaker cache starts empty")

        # Tier-3 warmup: prime trtllm-serve with realistic-shape requests.
        # Eliminates ~1s cold start (N=1 p99>1200ms tail).
        # 3 separate warmup calls with varying shapes to trigger multiple TRT kernel
        # specializations (prefill prefix sizes, generate lengths).
        # Tier-A A3: ALSO warm prefix-cache for each cached speaker — first real
        # request with that speaker hits trtllm-serve's block_reuse cache,
        # skipping prefill entirely (cold N=1 ~1012 ms → ~280 ms).
        warmup_lock = "/tmp/.bls_llm_warmup_done"
        if not os.path.exists(warmup_lock):
            try:
                import httpx as _httpx_sync
                # Build a realistic-sized assistant content (25 speech tokens like our prod prompts)
                spk_tokens_sample = "".join(f"<|s_{i*7 % 6500}|>" for i in range(25))
                ref_text = "You are a helpful assistant.<|endofprompt|>Hello world, this is a warmup."
                self.logger.log_info("Tier-3: warming up trtllm-serve (3 dummy requests)...")
                t0 = time.time()
                for i, max_tok in enumerate([10, 30, 100]):  # vary gen length
                    payload = {
                        "model": "trt_engines_bfloat16",
                        "messages": [
                            {"role": "user", "content": ref_text + f" Variant {i}."},
                            {"role": "assistant", "content": spk_tokens_sample},
                        ],
                        "max_tokens": max_tok,
                        "temperature": 0.8,
                        "stream": False,
                    }
                    resp = _httpx_sync.post(self.api_base, json=payload, timeout=60.0)
                    resp.raise_for_status()
                # Tier-A A3: prefix-cache prewarm for each cached speaker.
                # One short request per speaker, identical prefix to what real
                # requests will use → trtllm block_reuse cache populated.
                for spk_name, cache_key in self.speaker_name_to_cache_key.items():
                    cached = self.speaker_cache.get(cache_key)
                    if cached is None:
                        continue
                    spk_str = cached.get("prompt_speech_tokens_str")
                    if spk_str is None:
                        spk_str = self._convert_speech_tokens_to_str(cached["prompt_speech_tokens_for_llm"])
                    payload = {
                        "model": "trt_engines_bfloat16",
                        "messages": [
                            {"role": "user", "content": f"{cache_key}Warmup."},
                            {"role": "assistant", "content": spk_str},
                        ],
                        "max_tokens": 8,
                        "temperature": 0.8,
                        "stream": False,
                    }
                    resp = _httpx_sync.post(self.api_base, json=payload, timeout=60.0)
                    resp.raise_for_status()
                    self.logger.log_info(f"  prefix-cache warmed for speaker '{spk_name}'")
                with open(warmup_lock, "w") as f:
                    f.write(f"warmed at {time.time():.0f}")
                self.logger.log_info(f"LLM warmup OK in {time.time()-t0:.2f}s")
            except Exception as e:
                self.logger.log_warn(f"LLM warmup failed (continuing): {e}")

    def _cache_touch(self, key):
        """Mark cache entry as recently-used (LRU move-to-end)."""
        if key in self.speaker_cache:
            self.speaker_cache.move_to_end(key)

    def _cache_put(self, key, value):
        """Insert into cache with LRU eviction (Tier-1 C3)."""
        if key in self.speaker_cache:
            self.speaker_cache.move_to_end(key)
            self.speaker_cache[key] = value
            return
        self.speaker_cache[key] = value
        # Evict oldest entries beyond max_size, free GPU tensors explicitly.
        while len(self.speaker_cache) > self.cache_max_size:
            old_key, old_val = self.speaker_cache.popitem(last=False)
            # Don't evict named speakers from spk2info.pt
            if old_key in self.speaker_name_to_cache_key.values():
                self.speaker_cache[old_key] = old_val  # put back
                self.speaker_cache.move_to_end(old_key, last=False)
                break  # we wrapped around — stop eviction
            old_val.pop('prompt_speech_feat', None)
            old_val.pop('prompt_spk_embedding', None)
            # Discard lock if no concurrent waiter
            self._cache_locks.pop(old_key, None)

    def _cache_lock(self, key):
        """Get or create per-key async lock (Tier-1 C2)."""
        lock = self._cache_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._cache_locks[key] = lock
        return lock

    def _convert_speech_tokens_to_str(self, speech_tokens):
        """Convert speech token IDs tensor/list to string like '<|s_N|>'."""
        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens = speech_tokens.cpu().numpy().flatten().tolist()
        return "".join(f"<|s_{int(tid)}|>" for tid in speech_tokens)

    def _get_cached_prompt_str(self, cache_key, prompt_speech_tokens):
        """Tier-A A1: return baked LLM token string from spk2info cache if
        available, else fallback to runtime conversion. Saves ~1-3 ms per
        request on cached-speaker hot path (was 252 string formats per call)."""
        if cache_key in self.speaker_cache:
            cached = self.speaker_cache[cache_key]
            if 'prompt_speech_tokens_str' in cached:
                return cached['prompt_speech_tokens_str']
        return self._convert_speech_tokens_to_str(prompt_speech_tokens)

    def _extract_speech_feat(self, speech):
        """Extract mel spectrogram from 24kHz speech for flow prompt."""
        speech_feat = mel_spectrogram(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0).to(self.device)
        return speech_feat

    async def forward_llm_streaming(self, target_text, reference_text, prompt_speech_tokens):
        """Async generator: stream LLM tokens via httpx SSE."""
        full_text = f"{reference_text}{target_text}"
        # Tier-A A1: cached str lookup (hot path optimization)
        prompt_speech_tokens_str = self._get_cached_prompt_str(reference_text, prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        # Tier-A A6: max_tokens 400→200. Voice-chat texts produce ~50-150 speech
        # tokens. 200 is well above typical, but smaller than 400 → less LLM
        # scheduler memory allocation overhead. Defensive cap still applies.
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            "max_tokens": 200,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": True,
        }

        # Tier-1 H4: pre-compiled regex (_TOKEN_RE) + start-index tracking
        # instead of per-line re.search + buffer = buffer[match.end():] (full string copy).
        # On long streams (100+ tokens) saves ~1-3 ms parse overhead.
        buffer = ""
        scan_pos = 0
        async with self.http_client.stream("POST", self.api_base, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                line_data = line[6:].strip()
                if line_data == "[DONE]":
                    break
                try:
                    json_data = json.loads(line_data)
                except json.JSONDecodeError:
                    continue
                content = json_data.get("choices", [{}])[0].get("delta", {}).get("content")
                if not content:
                    continue
                buffer += content
                # Scan forward from where we last stopped — avoids O(N^2) on long buffers.
                for m in _TOKEN_RE.finditer(buffer, scan_pos):
                    yield int(m.group(1))
                    scan_pos = m.end()
                # Periodically truncate buffer to keep it small (every ~256 chars consumed)
                if scan_pos > 256:
                    buffer = buffer[scan_pos:]
                    scan_pos = 0

        # Flush remaining tokens after stream end
        for m in _TOKEN_RE.finditer(buffer, scan_pos):
            yield int(m.group(1))

    async def forward_llm_offline(self, target_text, reference_text, prompt_speech_tokens):
        """Non-streaming LLM call, returns all speech token IDs at once."""
        full_text = f"{reference_text}{target_text}"
        # Tier-A A1: cached str lookup
        prompt_speech_tokens_str = self._get_cached_prompt_str(reference_text, prompt_speech_tokens)

        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_tokens_str}
        ]
        payload = {
            "model": "trt_engines_bfloat16",
            "messages": chat,
            "max_tokens": 400,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<|eos1|>", "<|eos|>"],
            "stream": False,
        }
        response = await self.http_client.post(self.api_base, json=payload, timeout=None)
        response.raise_for_status()
        response_json = response.json()
        generated_content = response_json['choices'][0]['message']['content']
        speech_ids = parse_speech_token_string(generated_content)
        # return [sid + ORIGINAL_VOCAB_SIZE for sid in speech_ids]
        return speech_ids

    async def forward_audio_tokenizer(self, wav, wav_len):
        """Async BLS call to audio_tokenizer (parallelizable with speaker_embedding)."""
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['prompt_speech_tokens'],
            inputs=[wav, wav_len]
        )
        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_speech_tokens = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_speech_tokens')
        return torch.utils.dlpack.from_dlpack(prompt_speech_tokens.to_dlpack()).cpu()

    async def forward_speaker_embedding(self, wav):
        """Async BLS call to speaker_embedding (parallelizable with audio_tokenizer)."""
        inference_request = pb_utils.InferenceRequest(
            model_name='speaker_embedding',
            requested_output_names=['prompt_spk_embedding'],
            inputs=[pb_utils.Tensor.from_dlpack("reference_wav", to_dlpack(wav))]
        )
        inference_response = await inference_request.async_exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        prompt_spk_embedding = pb_utils.get_output_tensor_by_name(
            inference_response, 'prompt_spk_embedding')
        return torch.utils.dlpack.from_dlpack(prompt_spk_embedding.to_dlpack())

    def _ensure_t2w_dispatcher(self):
        """Lazy-start the background coordinator task (initialize is sync, can't)."""
        if self._t2w_dispatcher_task is None or self._t2w_dispatcher_task.done():
            self._t2w_dispatcher_task = asyncio.create_task(self._t2w_dispatcher())

    @staticmethod
    def _t2w_shape_key(item):
        return (item['target'].shape[1], item['prompt'].shape[1],
                item['pfeat'].shape[1], item['finalize'])

    async def _t2w_dispatcher(self):
        """Background: drain queue, group by shape, dispatch batched.

        Strategy: blocking await for first item, optionally sleep `_t2w_wait_ms`
        to let other concurrent streams accumulate, then opportunistic drain.

        IMPORTANT LIMITATION: `flow.inference_batched()` asserts uniform
        prompt_token_len/token_len/prompt_feat_len across the batch. Two streams
        can only batch when (a) they're at the same chunk_index AND (b) using
        same speaker (same prompt). For production traffic with a small set of
        cached speakers, this engages naturally. For test data with N distinct
        references, batching essentially never engages — `_t2w_shape_key`
        partition guarantees correctness but yields singleton groups.
        """
        try:
            while True:
                first = await self._t2w_queue.get()
                if self._t2w_wait_ms > 0:
                    await asyncio.sleep(self._t2w_wait_ms / 1000.0)
                batch = [first]
                first_key = self._t2w_shape_key(first)
                deferred = []
                while not self._t2w_queue.empty() and len(batch) < self._t2w_max_batch:
                    try:
                        item = self._t2w_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if self._t2w_shape_key(item) == first_key:
                        batch.append(item)
                    else:
                        deferred.append(item)
                for item in deferred:
                    self._t2w_queue.put_nowait(item)
                self._t2w_dispatch_count += 1
                if len(batch) > 1:
                    self.logger.log_info(
                        f"[inst {self._instance_id}] t2w batched B={len(batch)} "
                        f"key={first_key}")
                asyncio.create_task(self._t2w_dispatch_batch(batch))
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.logger.log_error(f"t2w_dispatcher crashed: {e}")

    async def _t2w_dispatch_batch(self, batch):
        """Send ONE batched InferenceRequest, slice output, fulfill futures."""
        try:
            B = len(batch)
            target_batch = torch.cat([it['target'] for it in batch], dim=0)
            prompt_batch = torch.cat([it['prompt'] for it in batch], dim=0)
            pfeat_batch  = torch.cat([it['pfeat']  for it in batch], dim=0)
            spk_batch    = torch.cat([it['spk']    for it in batch], dim=0)
            finalize_val = batch[0]['finalize']

            target_pb = pb_utils.Tensor.from_dlpack(
                "target_speech_tokens", to_dlpack(target_batch))
            prompt_pb = pb_utils.Tensor.from_dlpack(
                "prompt_speech_tokens", to_dlpack(prompt_batch))
            pfeat_pb  = pb_utils.Tensor.from_dlpack(
                "prompt_speech_feat", to_dlpack(pfeat_batch))
            spk_pb    = pb_utils.Tensor.from_dlpack(
                "prompt_spk_embedding", to_dlpack(spk_batch))
            # Scalar finalize: same for all B items in a homogeneous batch.
            # Use shape [B, 1] (Triton expects batch dim).
            finalize_pb = pb_utils.Tensor("finalize",
                np.array([[finalize_val]] * B, dtype=np.bool_))
            # token_offset is per-request; we slice in BLS after receiving mel.
            # Use 0 dummy to make token2wav happy (slicing in token2wav is no-op for 0).
            tok_off_pb = pb_utils.Tensor("token_offset",
                np.array([[0]] * B, dtype=np.int32))

            inputs = [target_pb, prompt_pb, pfeat_pb, spk_pb, tok_off_pb, finalize_pb]
            ir = pb_utils.InferenceRequest(
                model_name='token2wav',
                requested_output_names=['mel'],
                inputs=inputs,
            )
            resp = await ir.async_exec()
            if resp.has_error():
                raise pb_utils.TritonModelException(resp.error().message())

            mel_t = pb_utils.get_output_tensor_by_name(resp, 'mel')
            mel = torch.utils.dlpack.from_dlpack(mel_t.to_dlpack())
            # mel shape: [B, 80, T_mel] (or [1, 80, T_mel] if token2wav still returns single)
            if mel.dim() == 3 and mel.shape[0] == 1 and B > 1:
                # token2wav returned a single squeezed mel — shouldn't happen with batched
                # input, but defensive split if it does. Split equally.
                pass  # leave as-is, downstream will fail informatively

            for i, it in enumerate(batch):
                if mel.dim() == 3:
                    row_mel = mel[i] if mel.shape[0] >= B else mel[0]  # [80, T_mel]
                else:
                    row_mel = mel  # already [80, T]
                # Apply per-request token_offset slicing
                if it['tok_off'] is not None:
                    row_mel = row_mel[:, it['tok_off'] * self.token_mel_ratio:]
                if not it['future'].done():
                    it['future'].set_result(row_mel)
        except Exception as e:
            for it in batch:
                if not it['future'].done():
                    it['future'].set_exception(e)

    async def forward_token2wav(self, target_speech_tokens, prompt_speech_tokens,
                                prompt_speech_feat, prompt_spk_embedding,
                                request_id, token_offset=None, finalize=True,
                                priority=100, model_name='token2wav'):
        """Tier-4 coordinator path: queue request, await batched dispatch.

        Multiple concurrent streams in the same BLS event loop accumulate naturally
        in the queue while the dispatcher is awaiting previous batch's TRT call.
        Real GPU batching happens without artificial queue_delay.
        """
        self._ensure_t2w_dispatcher()
        future = asyncio.get_event_loop().create_future()
        self._t2w_push_count += 1
        await self._t2w_queue.put({
            'target': target_speech_tokens,
            'prompt': prompt_speech_tokens,
            'pfeat':  prompt_speech_feat,
            'spk':    prompt_spk_embedding,
            'tok_off': token_offset,
            'finalize': bool(finalize),
            'future': future,
        })
        return await future

    def _ensure_voc_dispatcher(self):
        if self._voc_dispatcher_task is None or self._voc_dispatcher_task.done():
            self._voc_dispatcher_task = asyncio.create_task(self._voc_dispatcher())

    @staticmethod
    def _voc_shape_key(item):
        # mel: [1, 80, T_mel] — only T_mel + finalize matters
        return (item['mel'].shape[2], item['finalize'])

    async def _voc_dispatcher(self):
        """BLS-side coordinator for vocoder. Same opportunistic group-by-shape
        pattern as t2w (see _t2w_dispatcher). Engages naturally when streams
        in same BLS event loop are at same mel-length, which requires same
        prompt + same chunk_index.
        """
        try:
            while True:
                first = await self._voc_queue.get()
                if self._voc_wait_ms > 0:
                    await asyncio.sleep(self._voc_wait_ms / 1000.0)
                batch = [first]
                first_key = self._voc_shape_key(first)
                deferred = []
                while not self._voc_queue.empty() and len(batch) < self._voc_max_batch:
                    try:
                        item = self._voc_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if self._voc_shape_key(item) == first_key:
                        batch.append(item)
                    else:
                        deferred.append(item)
                for item in deferred:
                    self._voc_queue.put_nowait(item)
                self._voc_dispatch_count += 1
                if len(batch) > 1:
                    self.logger.log_info(
                        f"[inst {self._instance_id}] voc batched B={len(batch)} "
                        f"key={first_key}")
                asyncio.create_task(self._voc_dispatch_batch(batch))
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.logger.log_error(f"voc_dispatcher crashed: {e}")

    async def _voc_dispatch_batch(self, batch):
        """Stack mel tensors, send ONE batched InferenceRequest, slice output."""
        try:
            B = len(batch)
            # Each item['mel'] shape [1, 80, T] (uniform T guaranteed by shape_key)
            mel_batch = torch.cat([it['mel'] for it in batch], dim=0)  # [B, 80, T]
            finalize_val = batch[0]['finalize']
            mel_pb = pb_utils.Tensor.from_dlpack("mel", to_dlpack(mel_batch.float()))
            finalize_pb = pb_utils.Tensor("finalize",
                np.array([[finalize_val]] * B, dtype=np.bool_))
            ir = pb_utils.InferenceRequest(
                model_name='vocoder',
                requested_output_names=['tts_speech'],
                inputs=[mel_pb, finalize_pb],
            )
            resp = await ir.async_exec()
            if resp.has_error():
                raise pb_utils.TritonModelException(resp.error().message())
            sp = pb_utils.get_output_tensor_by_name(resp, 'tts_speech')
            sp_t = torch.utils.dlpack.from_dlpack(sp.to_dlpack())
            # Expected: [B, T_audio]. If degenerate squeezed to [T_audio], unsqueeze.
            if sp_t.dim() == 1:
                sp_t = sp_t.unsqueeze(0)
            sp_cpu = sp_t.cpu()
            for i, it in enumerate(batch):
                row = sp_cpu[i] if sp_cpu.shape[0] >= B else sp_cpu[0]
                row = row.unsqueeze(0) if row.dim() == 1 else row  # [1, T]
                if not it['future'].done():
                    it['future'].set_result(row)
        except Exception as e:
            for it in batch:
                if not it['future'].done():
                    it['future'].set_exception(e)

    async def forward_vocoder(self, mel, finalize):
        """Async BLS call to vocoder via in-process coordinator (Tier-4 batching)."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [80, T] -> [1, 80, T]
        self._ensure_voc_dispatcher()
        self._voc_push_count += 1
        future = asyncio.get_event_loop().create_future()
        await self._voc_queue.put({
            'mel': mel,
            'finalize': bool(finalize),
            'future': future,
        })
        return await future

    def _apply_instruction_override(self, request, reference_text):
        """If request has `instruction` input, replace the baked instruction
        prefix in reference_text (format: "{instr}<|endofprompt|>{transcription}").
        For cached speakers this lets one speaker produce multiple speaking
        styles without rebaking spk2info. No-op when input absent or empty.
        """
        try:
            instr_t = pb_utils.get_input_tensor_by_name(request, "instruction")
        except Exception:
            return reference_text
        if instr_t is None:
            return reference_text
        user_instr = instr_t.as_numpy()[0][0].decode("utf-8").strip()
        if not user_instr:
            return reference_text
        if "<|endofprompt|>" in reference_text:
            _, transcription = reference_text.split("<|endofprompt|>", 1)
            return f"{user_instr}<|endofprompt|>{transcription}"
        return f"{user_instr}<|endofprompt|>{reference_text}"

    async def _prepare_prompt(self, request):
        """Extract reference audio, tokenize, compute speaker embedding and mel feat.

        Tier-1 changes:
        - C2: per-key async lock prevents duplicate work on concurrent first-time
          requests with the same reference_text.
        - C3: LRU put + touch on hit (capped GPU memory).
        - M4: cached resampler (built once in initialize).
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
                self._cache_touch(cache_key)
                return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                        cached['prompt_speech_feat'], cached['prompt_spk_embedding'], cache_key)

        wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
        wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")

        reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text")
        reference_text = reference_text.as_numpy()[0][0].decode('utf-8') if reference_text is not None else ""
        if '<|endofprompt|>' not in reference_text:
            reference_text = 'You are a helpful assistant.<|endofprompt|>' + reference_text

        # Fast path: cache hit (no lock needed for read)
        if reference_text in self.speaker_cache:
            cached = self.speaker_cache[reference_text]
            self._cache_touch(reference_text)
            return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                    cached['prompt_speech_feat'], cached['prompt_spk_embedding'], reference_text)

        # No reference audio — fallback to default speaker
        if wav is None and self.default_speaker_key is not None:
            cached = self.speaker_cache[self.default_speaker_key]
            self._cache_touch(self.default_speaker_key)
            return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                    cached['prompt_speech_feat'], cached['prompt_spk_embedding'],
                    self.default_speaker_key)

        if wav is None:
            raise pb_utils.TritonModelException(
                "No reference_wav provided and no spk2info.pt loaded.")

        # Slow path: compute under per-key lock — concurrent requests with same
        # reference_text will wait for first to populate cache (Tier-1 C2).
        async with self._cache_lock(reference_text):
            # Re-check cache after acquiring lock (another coroutine may have filled it)
            if reference_text in self.speaker_cache:
                cached = self.speaker_cache[reference_text]
                self._cache_touch(reference_text)
                return (cached['prompt_speech_tokens_for_llm'], cached['prompt_speech_tokens'],
                        cached['prompt_speech_feat'], cached['prompt_spk_embedding'], reference_text)

            # Audio tokenizer + speaker embedding in parallel (L40S round-5 step 10)
            wav_np = wav.as_numpy()
            wav_len_val = wav_len.as_numpy()[0][0]
            wav_tensor = torch.from_numpy(wav_np)
            wav_tensor = wav_tensor[:, :wav_len_val]

            prompt_speech_tokens, prompt_spk_embedding = await asyncio.gather(
                self.forward_audio_tokenizer(wav, wav_len),
                self.forward_speaker_embedding(wav_tensor),
            )
            prompt_speech_tokens = prompt_speech_tokens.unsqueeze(0)

            # Tier-1 M4: cached resampler instead of constructing new every call
            prompt_speech_resample = self.resampler_16to24(wav_tensor)
            speech_feat = self._extract_speech_feat(prompt_speech_resample)

            prompt_speech_tokens_for_llm = prompt_speech_tokens.clone()
            token_len = min(int(speech_feat.shape[1] / 2), prompt_speech_tokens.shape[-1])
            prompt_speech_feat = speech_feat[:, :2 * token_len].contiguous().half()
            prompt_speech_tokens = prompt_speech_tokens[:, :token_len].contiguous()

            # Tier-1 C3: LRU put (caps memory at cache_max_size)
            self._cache_put(reference_text, {
                'prompt_speech_tokens_for_llm': prompt_speech_tokens_for_llm,
                'prompt_speech_tokens': prompt_speech_tokens,
                'prompt_speech_feat': prompt_speech_feat,
                'prompt_spk_embedding': prompt_spk_embedding,
            })

        return prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, prompt_spk_embedding, reference_text

    async def _process_request_streaming(self, request):
        """Process a single request in streaming (decoupled) mode.

        Tier-2 A5: gated by self._stream_semaphore to bound concurrent in-flight
        streams per BLS instance. Excess requests wait → graceful degradation
        instead of catastrophic tail.
        """
        request_id = request.request_id()
        response_sender = request.get_response_sender()

        # Tier-2 A5: bounded concurrency (explicit acquire/release to avoid indent shift)
        await self._stream_semaphore.acquire()
        try:
            prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
                prompt_spk_embedding, reference_text = await self._prepare_prompt(request)
            reference_text = self._apply_instruction_override(request, reference_text)

            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')

            # Tier-1 H1: pre-allocate GPU tensor for all LLM tokens (max 400 by default).
            # Was: torch.tensor(list)[:end_idx].to(device) rebuilt fresh every chunk → O(N²).
            # Now: scalar copy into pre-allocated buffer + zero-copy slice for chunk view.
            MAX_LLM_TOKENS = 400
            all_tokens_gpu = torch.zeros(1, MAX_LLM_TOKENS, dtype=torch.int32, device=self.device)
            tokens_count = 0
            token_offset = 0
            chunk_index = 0
            this_token_hop_len = self.token_hop_len
            speech_offset = 0
            start_time = time.time()

            # Tier-2 H2: pre-allocated mel buffer. Was:
            #   accumulated_mel = torch.cat([accumulated_mel, mel_chunk]) — O(N²) memory churn,
            #   on chunk #5 copies 5× audio length. Pre-allocate full size, slice instead.
            # mel buffer = MAX_LLM_TOKENS × token_mel_ratio = 800 frames (~16s @ 50Hz).
            MAX_MEL_FRAMES = MAX_LLM_TOKENS * self.token_mel_ratio
            accumulated_mel = torch.zeros(1, 80, MAX_MEL_FRAMES,
                                         dtype=torch.float32, device=self.device)
            mel_len = 0  # current valid length in accumulated_mel

            async for generated_id in self.forward_llm_streaming(
                target_text=target_text,
                reference_text=reference_text,
                prompt_speech_tokens=prompt_speech_tokens_for_llm,
            ):
                if tokens_count >= MAX_LLM_TOKENS:
                    # Defensive: LLM exceeded expected max_tokens; expand buffer 2×
                    new_buf = torch.zeros(1, MAX_LLM_TOKENS * 2, dtype=torch.int32, device=self.device)
                    new_buf[:, :tokens_count] = all_tokens_gpu[:, :tokens_count]
                    all_tokens_gpu = new_buf
                    MAX_LLM_TOKENS *= 2
                all_tokens_gpu[0, tokens_count] = generated_id
                tokens_count += 1

                while True:
                    pending_num = tokens_count - token_offset
                    if pending_num < this_token_hop_len + self.flow_pre_lookahead_len:
                        break

                    end_idx = token_offset + this_token_hop_len + self.flow_pre_lookahead_len
                    # Zero-copy slice on pre-allocated GPU buffer
                    this_tokens = all_tokens_gpu[:, :end_idx].contiguous()

                    mel_chunk = await self.forward_token2wav(
                        this_tokens, prompt_speech_tokens,
                        prompt_speech_feat, prompt_spk_embedding,
                        request_id, token_offset=token_offset, finalize=False,
                        priority=chunk_index + 1,
                    )

                    # Tier-2 H2: in-place append to pre-allocated buffer
                    if mel_chunk.dim() == 2:
                        mel_chunk = mel_chunk.unsqueeze(0)
                    chunk_T = mel_chunk.shape[2]
                    if mel_len + chunk_T > MAX_MEL_FRAMES:
                        # Defensive grow 2× if mel buffer overflows
                        new_buf = torch.zeros(1, 80, MAX_MEL_FRAMES * 2,
                                              dtype=torch.float32, device=self.device)
                        new_buf[:, :, :mel_len] = accumulated_mel[:, :, :mel_len]
                        accumulated_mel = new_buf
                        MAX_MEL_FRAMES *= 2
                    accumulated_mel[:, :, mel_len:mel_len + chunk_T] = mel_chunk.to(torch.float32)
                    mel_len += chunk_T

                    # Vocoder consumes valid region (contiguous() since slice is a view)
                    speech = await self.forward_vocoder(
                        accumulated_mel[:, :, :mel_len].contiguous(), finalize=False)

                    # Extract new speech
                    new_speech = speech[:, speech_offset:]
                    speech_offset += new_speech.shape[1]

                    if new_speech.shape[1] > 0:
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
                                next_pending = tokens_count - token_offset
                                if multiples > 4:
                                    this_token_hop_len = (next_pending // self.token_hop_len + 1) * self.token_hop_len
                                elif multiples > 2:
                                    this_token_hop_len = (next_pending // self.token_hop_len) * self.token_hop_len
                                else:
                                    this_token_hop_len = self.token_hop_len
                                this_token_hop_len = max(self.token_hop_len, this_token_hop_len)

                    chunk_index += 1

            # Final chunk with remaining tokens (Tier-1 H1: zero-copy slice)
            if tokens_count > 0:
                remaining_tokens = all_tokens_gpu[:, :tokens_count].contiguous()

                mel_chunk = await self.forward_token2wav(
                    remaining_tokens, prompt_speech_tokens,
                    prompt_speech_feat, prompt_spk_embedding,
                    request_id, token_offset=token_offset, finalize=True,
                    priority=chunk_index + 1,
                )

                # Tier-2 H2: append final chunk to pre-allocated buffer
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
                if new_speech.shape[1] > 0:
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
        finally:
            # Tier-2 A5: release semaphore slot
            self._stream_semaphore.release()

    async def _process_request_offline(self, request):
        """Process a single request in offline (non-decoupled) mode."""
        request_id = request.request_id()

        prompt_speech_tokens_for_llm, prompt_speech_tokens, prompt_speech_feat, \
            prompt_spk_embedding, reference_text = await self._prepare_prompt(request)
        reference_text = self._apply_instruction_override(request, reference_text)

        target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
        target_text = target_text[0][0].decode('utf-8')

        # Get all speech tokens at once (use full untruncated prompt tokens for LLM)
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

        audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(speech))
        return pb_utils.InferenceResponse(output_tensors=[audio_tensor])

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
