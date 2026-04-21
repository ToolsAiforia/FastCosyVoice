#!/usr/bin/env python3
"""Generate spk2info.pt for CosyVoice3 speaker cache.

This script pre-computes audio tokens, speaker embeddings, and mel features
from reference audio files so that the Triton BLS model can skip these
expensive steps at inference time.

Usage (inside the container):
    python3 generate_spk2info.py \
        --model-dir ./Fun-CosyVoice3-0.5B-2512 \
        --audio ./Emily.wav \
        --reference-text "So my favorite podcast at the moment..." \
        --speaker-name emily \
        --output ./Fun-CosyVoice3-0.5B-2512/spk2info.pt

    # Add more speakers to existing file:
    python3 generate_spk2info.py \
        --model-dir ./Fun-CosyVoice3-0.5B-2512 \
        --audio ./Bob.wav \
        --reference-text "Hello this is Bob speaking..." \
        --speaker-name bob \
        --output ./Fun-CosyVoice3-0.5B-2512/spk2info.pt
"""
import argparse
import os

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import s3tokenizer
import onnxruntime
from functools import partial
from matcha.utils.audio import mel_spectrogram as matcha_mel_spectrogram


# CosyVoice3 mel params
mel_spectrogram = partial(
    matcha_mel_spectrogram,
    n_fft=1920, num_mels=80, sampling_rate=24000,
    hop_size=480, win_size=1920, fmin=0, fmax=None, center=False,
)


def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform  # [1, T]


def compute_speech_tokens(waveform_16k, tokenizer_model):
    """Run s3tokenizer on 16kHz waveform → token IDs."""
    wav = waveform_16k.squeeze(0)  # [T]
    mel = s3tokenizer.log_mel_spectrogram(wav)
    mels, mels_lens = s3tokenizer.padding([mel])
    device = next(tokenizer_model.parameters()).device
    codes, codes_lens = tokenizer_model.quantize(
        mels.to(device), mels_lens.to(device)
    )
    tokens = codes[0, : codes_lens[0].item()].cpu()
    return tokens  # [T_tokens]


def compute_speaker_embedding(waveform_16k, spk_model_path):
    """Run CAMPPlus on 16kHz waveform → speaker embedding."""
    feat = kaldi.fbank(waveform_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = feat - feat.mean(dim=0, keepdim=True)

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    spk_session = onnxruntime.InferenceSession(
        spk_model_path, sess_options=option, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    embedding = spk_session.run(
        None, {spk_session.get_inputs()[0].name: spk_feat.unsqueeze(0).numpy()}
    )[0]
    return torch.tensor(embedding).half()  # [1, 192]


def compute_speech_feat(waveform_16k):
    """Resample to 24kHz and compute mel spectrogram."""
    waveform_24k = torchaudio.transforms.Resample(16000, 24000)(waveform_16k)
    speech_feat = mel_spectrogram(waveform_24k).squeeze(0).transpose(0, 1)
    return speech_feat.unsqueeze(0)  # [1, T_feat, 80]


def build_spk_info(waveform_16k, tokenizer_model, campplus_path):
    """Compute all speaker info tensors from 16kHz waveform."""
    # 1. Audio tokens
    prompt_speech_tokens = compute_speech_tokens(waveform_16k, tokenizer_model)
    prompt_speech_tokens = prompt_speech_tokens.unsqueeze(0)  # [1, T_tokens]
    prompt_speech_tokens_for_llm = prompt_speech_tokens.clone()

    # 2. Speaker embedding
    prompt_spk_embedding = compute_speaker_embedding(waveform_16k, campplus_path)

    # 3. Mel features
    speech_feat = compute_speech_feat(waveform_16k)

    # 4. Align tokens and feat to 2:1 ratio (same logic as BLS _prepare_prompt)
    token_len = min(int(speech_feat.shape[1] / 2), prompt_speech_tokens.shape[-1])
    prompt_speech_feat = speech_feat[:, : 2 * token_len].contiguous().half()
    prompt_speech_tokens = prompt_speech_tokens[:, :token_len].contiguous()

    return {
        "prompt_speech_tokens_for_llm": prompt_speech_tokens_for_llm,
        "prompt_speech_tokens": prompt_speech_tokens,
        "prompt_speech_feat": prompt_speech_feat,
        "prompt_spk_embedding": prompt_spk_embedding,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate spk2info.pt for CosyVoice3")
    parser.add_argument("--model-dir", required=True, help="Path to CosyVoice3 model dir")
    parser.add_argument("--audio", required=True, help="Reference audio file (wav)")
    parser.add_argument("--reference-text", required=True, help="Transcript of reference audio")
    parser.add_argument("--speaker-name", required=True, help="Speaker identifier name")
    parser.add_argument("--output", required=True, help="Output spk2info.pt path")
    parser.add_argument("--device", default="cuda", help="Device for s3tokenizer")
    args = parser.parse_args()

    # Load models
    tokenizer_path = os.path.join(args.model_dir, "speech_tokenizer_v3.onnx")
    campplus_path = os.path.join(args.model_dir, "campplus.onnx")

    print(f"Loading s3tokenizer from {tokenizer_path}")
    tokenizer_model = s3tokenizer.load_model(tokenizer_path).to(args.device)

    # Load audio
    print(f"Loading audio from {args.audio}")
    waveform_16k = load_audio(args.audio, target_sr=16000)
    print(f"  Sample rate: 16000, Duration: {waveform_16k.shape[1] / 16000:.2f}s")

    # Compute speaker info
    print("Computing speaker info...")
    spk_info = build_spk_info(waveform_16k, tokenizer_model, campplus_path)

    # Build reference_text key (same prefix as BLS model)
    reference_text = args.reference_text
    if "<|endofprompt|>" not in reference_text:
        reference_text = "You are a helpful assistant.<|endofprompt|>" + reference_text
    spk_info["reference_text"] = reference_text

    # Load existing or create new
    if os.path.exists(args.output):
        print(f"Loading existing spk2info from {args.output}")
        spk2info = torch.load(args.output, map_location="cpu")
    else:
        spk2info = {}

    spk2info[args.speaker_name] = spk_info
    torch.save(spk2info, args.output)

    print(f"Saved speaker '{args.speaker_name}' to {args.output}")
    print(f"  reference_text key: {reference_text[:80]}...")
    print(f"  prompt_speech_tokens_for_llm: {spk_info['prompt_speech_tokens_for_llm'].shape}")
    print(f"  prompt_speech_tokens: {spk_info['prompt_speech_tokens'].shape}")
    print(f"  prompt_speech_feat: {spk_info['prompt_speech_feat'].shape}")
    print(f"  prompt_spk_embedding: {spk_info['prompt_spk_embedding'].shape}")
    print(f"  Total speakers in file: {list(spk2info.keys())}")


if __name__ == "__main__":
    main()
