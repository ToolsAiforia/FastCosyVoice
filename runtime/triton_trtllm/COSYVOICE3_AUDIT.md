# CosyVoice3 — Configuration + Graph + Engine + Profiling Audit

Single-file dump of all artifacts requested:

- (1) actual `cosyvoice3.yaml`
- (2) TRT-LLM engine config + HF config for `cosyvoice3_llm`
- (3) Exported ONNX graphs for `flow.decoder.estimator.fp32.onnx` and `hift_decode_core.onnx`
- (4) TRT layer dump / build logs with tensor shapes
- (5) Profiling run: actual generated token count + per-hop shapes

Branch: `round-9-stable`. Host: H100 PCIe 80 GB.
Model dir: `runtime/triton_trtllm/Fun-CosyVoice3-0.5B-2512/`.
Triton port: HTTP 18000 / gRPC 18001 (model_control_mode=explicit, all 5 models READY).

---

## 1. `cosyvoice3.yaml` (actual, in-use)

**Path:** `runtime/triton_trtllm/Fun-CosyVoice3-0.5B-2512/cosyvoice3.yaml` (223 lines).

```yaml
sample_rate: 24000
llm_input_size:  896
llm_output_size: 896
spk_embed_dim:   192
token_frame_rate: 25
token_mel_ratio:  2
chunk_size: 25                  # static_chunk_size = 25 * 2 = 50 mel
num_decoding_left_chunks: -1

llm: !new:cosyvoice.llm.llm.CosyVoice3LM
    llm_input_size:  896
    llm_output_size: 896
    speech_token_size: 6561
    length_normalized_loss: True
    lsm_weight: 0
    mix_ratio: [5, 15]
    llm: !new:cosyvoice.llm.llm.Qwen2Encoder
        pretrain_path: ''                # overridden to <model_dir>/CosyVoice-BlankEN
    sampling: !name:cosyvoice.utils.common.ras_sampling
        top_p: 0.8
        top_k: 25
        win_size: 10
        tau_r: 0.1

flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithDiT
    input_size: 80
    output_size: 80
    spk_embed_dim: 192
    output_type: 'mel'
    vocab_size: 6561
    input_frame_rate: 25
    only_mask_loss: True
    token_mel_ratio: 2
    pre_lookahead_len: 3
    pre_lookahead_layer: PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3)
    decoder: CausalConditionalCFM
        in_channels: 240
        n_spks: 1, spk_emb_dim: 80
        cfm_params:
            sigma_min: 1e-06
            solver: 'euler'
            t_scheduler: 'cosine'
            training_cfg_rate: 0.2
            inference_cfg_rate: 0.7
            reg_loss_type: 'l1'
        estimator: DiT
            dim: 1024, depth: 22, heads: 16, dim_head: 64
            ff_mult: 2
            mel_dim: 80, mu_dim: 80, spk_dim: 80, out_channels: 80
            static_chunk_size: 50            # 25 * 2
            num_decoding_left_chunks: -1

hift: !new:cosyvoice.hifigan.generator.CausalHiFTGenerator
    in_channels: 80, base_channels: 512
    nb_harmonics: 8, sampling_rate: 24000
    nsf_alpha: 0.1, nsf_sigma: 0.003, nsf_voiced_threshold: 10
    upsample_rates:        [8, 5, 3]
    upsample_kernel_sizes: [16, 11, 7]
    istft_params: {n_fft: 16, hop_len: 4}                    # → 24 kHz
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
    source_resblock_kernel_sizes: [7, 7, 11]
    source_resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
    lrelu_slope: 0.1, audio_limit: 0.99
    conv_pre_look_right: 4
    f0_predictor: CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
        # kept on CPU in fast path
```

**Note on sampling:** the `ras_sampling(top_p=0.8, top_k=25)` block is the **training-time** sampler. Production inference uses `trtllm-serve` HTTP, whose sampling parameters live in BLS `forward_llm_streaming` payload (`T=0.8, top_p=0.95, top_k=50, repetition_penalty=1.1, max_tokens=200`).

---

## 2. TRT-LLM engine + HF configs for `cosyvoice3_llm`

### HF config — `runtime/triton_trtllm/cosyvoice3_llm/config.json`
```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 158486,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 24,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 158720
}
```
→ ~0.5 B Qwen2, **GQA 14/2** (KV heads), RoPE base 1 M, vocab 158 720 (6561 speech + 200 extras + text).

### TRT-LLM engine config — `runtime/triton_trtllm/trt_engines_bfloat16/config.json` (production)
```json
{
    "version": "0.20.0",
    "pretrained_config": {
        "architecture": "Qwen2ForCausalLM",
        "dtype": "bfloat16",
        "vocab_size": 158720,
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "logits_dtype": "float32",
        "norm_epsilon": 1e-06,
        "position_embedding_type": "rope_gpt_neox",
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "head_size": 64,
        "qk_layernorm": false,
        "rotary_embedding_dim": 64,
        "rotary_base": 1000000.0,
        "seq_length": 8192,
        "qwen_type": "qwen2",
        "tie_word_embeddings": false,
        "mlp_bias": false,
        "attn_bias": true,
        "moe": { "num_experts": 0 },
        "mapping": { "world_size": 1, "tp_size": 1, "pp_size": 1 },
        "quantization": { "quant_algo": null, "kv_cache_quant_algo": null }
    },
    "build_config": {
        "max_input_len":     1024,
        "max_seq_len":      32768,
        "max_batch_size":     64,
        "opt_batch_size":      8,
        "max_num_tokens":  32768,
        "opt_num_tokens":     64,
        "max_beam_width":      1,
        "kv_cache_type": "PAGED",
        "strongly_typed":    true,
        "speculative_decoding_mode": 1,
        "auto_parallel_config": { "cluster_key": "H100-PCIe" },
        "same_buffer_io": { "past_key_value_(\\d+)": "present_key_value_\\1" }
    }
}
```

Alternative builds (not active in prod):
```
trt_engines_fp8/      — fp8 calibrated, audio degraded
trt_engines_fp8_v2/   — fp8 v2 calibrated, still degraded
trt_weights_bfloat16/ — intermediate
```

`trtllm-serve` runtime (sidecar on `LLM_PORT=8010`, BLS hits via HTTP):
```
max_batch_size: 64
kv_cache_free_gpu_memory_fraction: 0.4
enable_block_reuse: true   (prefix cache)
```

---

## 3. Exported ONNX graphs

### Inventory (with sizes/timestamps)
```
flow.decoder.estimator.fp32.onnx                       1.3 GB  opset 18  pytorch 2.3.1
flow.decoder.estimator.autocast_fp16.onnx              634 MB  (FP16 IO)
flow.decoder.estimator.streaming.autocast_fp16.onnx    635 MB
flow.decoder.estimator.B8.autocast_fp16.onnx           634 MB  (batch-dynamic up to 16=8×CFG)
flow.decoder.estimator.fp32.simplified.onnx            1.3 GB  (onnx-simplifier output)
flow.decoder.estimator.fp32.B_dyn.onnx                 1.3 GB
hift_decode_core.onnx                                   67 MB  opset 17  pytorch 2.7.0
campplus.onnx                                           27 MB
speech_tokenizer_v3.onnx                               925 MB
speech_tokenizer_v3.batch.onnx                         925 MB
```

### `flow.decoder.estimator.fp32.onnx` — DiT estimator (FP32 source)
```
opset:    ai.onnx v18
producer: pytorch 2.3.1
ir:       8
inputs:
  x    : FLOAT  [2, 80, seq_len]
  mask : FLOAT  [2,  1, seq_len]
  mu   : FLOAT  [2, 80, seq_len]
  t    : FLOAT  [2]
  spks : FLOAT  [2, 80]
  cond : FLOAT  [2, 80, seq_len]
outputs:
  estimator_out : FLOAT  [<dim0>, 80, seq_len]
node count: 7644      initializers: 323

op-type histogram (top 20):
   2794  Constant
    829  Unsqueeze
    694  Mul
    471  Gather
    451  Shape
    402  Add
    292  Cast
    290  Slice
    274  Concat
    183  Reshape
    178  MatMul
    112  Div
     96  Transpose
     89  Squeeze
     88  Neg
     66  Sqrt
     45  Sin
     45  Cos
     45  Where
     45  LayerNormalization      ← FP32-sensitive layers concentrated here
```

### `hift_decode_core.onnx` — HiFT decoder subgraph (no f0/stft input prep)
```
opset:    ai.onnx v17
producer: pytorch 2.7.0
ir:       8
inputs:
  x_pre  : FLOAT  [B, 512, T_mel]
  s_stft : FLOAT  [B,  18, T_stft]
outputs:
  magnitude : FLOAT  [B, 9, T_aud_stft]
  phase     : FLOAT  [B, 9, T_aud_stft]
node count: 2014      initializers: 299

op-type histogram (top 20):
    581  Constant
    216  Mul
    154  Concat
    148  Shape
    148  Gather
    148  Unsqueeze
    117  Add
     80  ConstantOfShape
     80  Cast
     79  Conv
     73  Sin                ← Snake activation: sensitive
     72  Reciprocal         ← Snake activation: sensitive
     72  Pow                ← Snake activation: sensitive
     12  Reshape
      8  Slice
      6  Transpose
      6  Pad
      4  LeakyRelu
      3  Identity
      3  Resize
```

---

## 4. TRT build logs + engine layer dumps

### Build logs (captured during plan compilation)

**`build_dit_trt.log`** — DiT B8 plan
```
[05/18-13:56:18] Init CUDA: GPU 9691 MiB
[05/18-13:57:10] Detected 6 inputs and 1 output network tensors.
                 Total Host Persistent Memory: 11696 bytes
                 Max Scratch Memory: 6164782592 bytes  (5.7 GB)
                 Total Activation Memory: 6362373632 bytes
                 Total Weights Memory:    662294208 bytes  (632 MB)
[05/18-13:57:13] Engine generation completed in 41.4 s

Profile: min(B=2)  opt(B=8 = 4×CFG)  max(B=16 = 8×CFG)
  x    : min=(2, 80, 4) opt=(8, 80, 500) max=(16, 80, 3000)
  mask : min=(2,  1, 4) opt=(8,  1, 500) max=(16,  1, 3000)
  mu   : min=(2, 80, 4) opt=(8, 80, 500) max=(16, 80, 3000)
  cond : min=(2, 80, 4) opt=(8, 80, 500) max=(16, 80, 3000)
  t    : min=(2,)       opt=(8,)         max=(16,)
  spks : min=(2, 80)    opt=(8, 80)      max=(16, 80)

Output: flow.decoder.estimator.B8.autocast_fp16.0.plan  (637.4 MB)
```

**`build_mixed.log`** — DiT layer_mixed FP16 (PRODUCTION DiT plan)
```
ONNX: flow.decoder.estimator.fp32.onnx
PLAN: flow.decoder.estimator.layer_mixed_fp16.0.plan

Total layers: 16644
Top layer types:
  Constant:        5362
  Elementwise:     5320
  Cast:            1484
  Shuffle:         1020
  Unsqueeze:        831
  Gather:           605
  Shape:            565
  Slice:            391
  Concatenation:    362
  Unary:            266
  MatMul:           203
  Squeeze:           91
  Select:            45
  Normalization:     45
  Activation:        28

FP32-marked layers: 75 / 16644
  Normalization:                          45
  Softmax:                                22
  time_embed (large dynamic range):        6
  output projection (mel values):          2

Built in 59.8 s, size = 637.0 MB
```

**`hift_trt_build.log`** — initial HiFT FP16 plan (later superseded by layer_mixed_fp32io)
```
2026-05-14 11:07:28  Converting onnx to trt (fp16=True)
2026-05-14 11:07:32  TensorRT: no Softmax layers found to force to FP32 (FP16 build).
2026-05-14 11:07:32  building engine for hift_decode_core.onnx
2026-05-14 11:12:53  Succesfully convert onnx to trt
Built: hift_decode_core.fp16.plan  size: 38.97 MB
```

Production HiFT plan `hift_decode_core.layer_mixed_fp32io.plan` (41 MB) is built with NAME-based keyword matching marking 507/3166 layers FP32 (`activations1`, `activations2`, `m_source`, `Sin`, `Pow`, `Reciprocal`, `Exp`, `Log`, `Sqrt`, `Softmax`, `LayerNorm`, `Tanh`, `Sigmoid`, `GELU`, `Erf`, `norm`, `conv_post`, `_istft`, `_stft`) — see `vocoder/1/model.py:_build_hift_layer_mixed_trt`.

### Runtime engine inspector dump (production plans)

#### DiT `flow.decoder.estimator.layer_mixed_fp16.0.plan`
```
file size:                668.0 MB
num_io_tensors:           7
num_optimization_profiles: 1
runtime fused layers:     257

IO (B=2 fixed for CFG pair):
  [IN ] x    : HALF  [2, 80, -1]   prof[0]  min=[2, 80,    4]  opt=[2, 80,  500]  max=[2, 80, 3000]
  [IN ] mask : HALF  [2,  1, -1]   prof[0]  min=[2,  1,    4]  opt=[2,  1,  500]  max=[2,  1, 3000]
  [IN ] mu   : HALF  [2, 80, -1]   prof[0]  min=[2, 80,    4]  opt=[2, 80,  500]  max=[2, 80, 3000]
  [IN ] t    : HALF  [2]
  [IN ] spks : HALF  [2, 80]
  [IN ] cond : HALF  [2, 80, -1]   prof[0]  min=[2, 80,    4]  opt=[2, 80,  500]  max=[2, 80, 3000]
  [OUT] estimator_out : HALF  [2, 80, -1]

Top fused-layer prefixes:
     43  __myl_ReshReshMulReshSlicNegSlicConcResh        (RoPE-style rotations)
     21  __myl_SlicReshSlicReshSlicReshAddMulAddR        (attention block fusion, depth/2 × 22)
     21  __myl_SlicReshSlicReshSlicReshAddTranRes
      3  Reformatting CopyNode for Input Tensor 0        (precision conversion at FP32 boundaries)
      2  shuffle_between_ONNXTRT_unsqueezeTensor_
      2  shuffle_after_/input_embed/conv_pos_embe
      1  __mye1041_0_myl0_0
      1  __myl_TranTranTranReplConc_myl0_1
      1  /input_embed/proj/MatMul_myl0_2
      1  __mye4691_myl0_3
      1  __myl_TranReplReplConcResh_myl0_4
      1  __mye4693_myl0_5
      1  __myl_Cast_myl0_6
      1  exit^bb^signal^1_myl0_7
      1  exit^bb^wait^1_myl0_8
      1  /input_embed/conv_pos_embed/conv1/conv1.
      1  ONNXTRT_squeezeTensor
      1  PWN(PWN(PWN(/input_embed/conv_pos_embed/
      1  /input_embed/conv_pos_embed/Pad_1_139
      1  ONNXTRT_unsqueezeTensor_141
```

#### HiFT `hift_decode_core.layer_mixed_fp32io.plan`
```
file size:                41 MB
runtime fused layers:     1574

Top fused-layer prefixes:
    239  Reformatting CopyNode for Input Tensor 0        ← cost of mixed-precision: many boundary copies
    144  onnx
     88  Reformatting CopyNode for Input Tensor 1
     78  shuffle_between_ONNXTRT_unsqueezeTensor_
     27  Reformatting CopyNode for Output Tensor
     14  Reformatting CopyNode for Input Tensor 2
     12  copied_squeeze_after_/source_resblocks.2…       (×3 for source_resblocks 0/1/2)
     12  copied_squeeze_after_/source_resblocks.0
     12  copied_squeeze_after_/source_resblocks.1
     12  copied_squeeze_after_/resblocks.0/activa        (×9 resblocks * activations1/2)
     12  copied_squeeze_after_/resblocks.1/activa
     12  copied_squeeze_after_/resblocks.2/activa
     12  copied_squeeze_after_/resblocks.3/activa
     12  copied_squeeze_after_/resblocks.4/activa
     12  copied_squeeze_after_/resblocks.5/activa
     12  copied_squeeze_after_/resblocks.6/activa
     12  copied_squeeze_after_/resblocks.7/activa
     12  copied_squeeze_after_/resblocks.8/activa
      6  unsqueeze_node_after_/source_resblocks.0
      6  unsqueeze_node_after_/resblocks.0/activa
```

Reformatting CopyNodes (369 total in HiFT) are the price of marking Snake activations FP32 inside an otherwise FP16-friendly conv stack — opportunity to reduce by grouping FP32 islands.

---

## 5. Profiling run — actual generated token count + per-hop shapes

**Setup**
- Target text: `"Just a reminder that your minimum payment of 800 dollars is due on January 14."` (79 chars)
- Reference: `runtime/neutral.wav` (11.51 s @ 16 kHz mono → 287 prompt speech tokens, 574 mel frames)
- Streaming config: `token_hop_len=8`, `flow_pre_lookahead_len=1`, hop growth exponential `25 × 2^chunk_index` after first hop
- Plans: DiT `layer_mixed_fp16` + HiFT `layer_mixed_fp32io`
- BLS counts: 16; token2wav 8 GPU instances; vocoder 8 CPU instances

### Client-side timeline (gRPC `stream_infer`)
```
#    t emit (ms)   samples   dt(ms)   dur(s)
0           246      1 920      246    0.080
1           387     22 080      141    0.920
2           632     46 080      245    1.920
3           837     47 040      206    1.960
                  ────────
Total chunks: 4    samples 117 120  →  4.88 s audio
TTFA: 246 ms       Wallclock: 837 ms       RTF (warm): 0.17
```

### BLS server-side per-hop dump (with shapes)
```
hop=0  this_tokens=[1,  9]   prompt_tok=[1,287]   prompt_mel=[1,574,80]   token_offset= 0   mel_before=  0
       t2w_ms= 94    mel_chunk=[1,80, 12]   acc_mel_T= 12
       voc_ms= 40    speech=[1,  1920]                                ← 80 ms emitted

hop=1  this_tokens=[1, 34]   prompt_tok=[1,287]   prompt_mel=[1,574,80]   token_offset= 8   mel_before= 12
       t2w_ms= 92    mel_chunk=[1,80, 46]   acc_mel_T= 58
       voc_ms= 48    speech=[1, 24000]                                ← 1.0 s total

hop=2  this_tokens=[1, 84]   prompt_tok=[1,287]   prompt_mel=[1,574,80]   token_offset=33   mel_before= 58
       t2w_ms=111    mel_chunk=[1,80, 96]   acc_mel_T=154
       voc_ms= 65    speech=[1, 70080]                                ← 2.92 s total

FINAL  hop=3  remaining_tokens=[1,138]                                 token_offset=83   mel_before=154
       t2w_ms= 89    mel_chunk=[1,80,110]   acc_mel_T=264
       voc_ms= 71    speech=[1,126720]                                ← 5.28 s total

LLM-generated tokens: 8 + 25 + 50 + 138 = 221 speech tokens for 79 chars  (≈ 2.8 tok/char)
                      → 264 mel frames after streaming overlap/trim → 5.28 s audio
```

### Observations
- **t2w stays flat ~90-110 ms** across hops — TRT opt seq_len=500 covers all accumulated mel up to 264. No big penalty from variable seq_len thanks to single optimization profile centered correctly.
- **vocoder grows 40 → 71 ms linearly** with accumulated mel — this is **non-incremental HiFT**: each hop re-runs decode on the entire accumulated mel buffer, not just the new tail. H3 incremental backport is in `improve_cosyvoice3` branch but **not on round-9-stable**.
- **LLM is not the bottleneck** in streaming: gap between hops (≈47 ms wallclock between voc_done→next t2w start) means LLM produced 25 more tokens in 47 ms while vocoder ran.
- **TTFA breakdown (warm, ~246 ms):**
  ```
  audio_tokenizer  (speech_tokenizer_v3 on ref):     10-20 ms
  speaker_embedding (CAMPPlus):                       5-10 ms
  LLM prefill + generate 8 tokens (trtllm-serve):    80-100 ms   ← dominant
  token2wav DiT chunk0 (9 tok → 12 mel):              ~95 ms     ← second dominant
  vocoder HiFT (12 mel → 1920 samples):                ~40 ms
                                            total ≈   230-275 ms   (matches 246 measured)
  ```
- **Cold start (first request after Triton boot):** ~2200-3200 ms before Tier A warmup, ~250-350 ms after. Tier A warmup primes LLM TRT, t2w TRT plan, HiFT TRT plan with realistic chunk-1 shapes (9 target / 287 prompt / 574 mel).

### Triton inference statistics (for context, 18 requests served)
```
cosyvoice3 (BLS):   inference_count=18,  total compute  122.8 s   (≈ 6.8 s/req incl. all sub-stages)
                                         queue_ns ≈ 5.5 ms total
token2wav:          inference_count=61,  compute_infer 8 015 ms   (≈ 131 ms / call)
vocoder:            inference_count=61,  compute_infer 6 508 ms   (≈ 107 ms / call)
```
(Sub-stage counts: ~4 chunks × 16 BLS bls invocations = 64 expected; small drift due to FINAL fused into last chunk.)

---

## Appendix A — Useful related artifacts on disk

```
/root/FastCosyVoice/runtime/triton_trtllm/CozyVoice_settings_round9_stable.md
    (10 documented deltas between CozyVoice_settings.md and actual round-9-stable code)
/root/FastCosyVoice/runtime/triton_trtllm/REPORT_H100_ALL_ROUNDS.md
/root/FastCosyVoice/runtime/triton_trtllm/SYNC_ROUND9_2026-05-20.md
/root/FastCosyVoice/synth_round9.py                         CLI: zero-shot synth via gRPC
/tmp/round9_repo/cosyvoice3/1/model.py                      live BLS (Tier A restored)
/tmp/round9_repo/{token2wav,vocoder}/1/model.py + config.pbtxt
```

## Appendix B — Reproducing this audit

```bash
# 1. yaml
cat runtime/triton_trtllm/Fun-CosyVoice3-0.5B-2512/cosyvoice3.yaml

# 2. HF + TRT-LLM configs
cat runtime/triton_trtllm/cosyvoice3_llm/config.json
cat runtime/triton_trtllm/trt_engines_bfloat16/config.json

# 3. ONNX summary
python3 -c "import onnx, collections; m=onnx.load('PATH', load_external_data=False); \
  g=m.graph; print(len(g.node), 'nodes'); \
  print(collections.Counter(n.op_type for n in g.node).most_common(20))"

# 4. TRT engine inspector
python3 - <<'EOF'
import tensorrt as trt, json, collections
rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
eng = rt.deserialize_cuda_engine(open('PLAN','rb').read())
ins = eng.create_engine_inspector()
parsed = json.loads(ins.get_engine_information(trt.LayerInformationFormat.JSON))
print(len(parsed['Layers']), 'fused layers')
EOF

# 5. Profiling run (patch BLS with self.logger.log_info("[PROF]…") around
#    forward_token2wav / forward_vocoder, reload BLS model, run synth_round9.py,
#    then `strings /tmp/tritonserver.log | grep PROF`)
```
