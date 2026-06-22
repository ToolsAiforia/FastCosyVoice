#!/usr/bin/env python3
"""Quality scorer for the CosyVoice3 re-optimization ladder.

Per-file + aggregate metrics for a set of synthesized wavs described by a manifest
(samples_*.json schema: {uid, text, raw_text, audio_path, system}). reference_name
is derived from `system` (strip cosy3_/vanilla_/neutral2_ prefix) and resolved to
reference_samples/<name>_16k.wav for speaker similarity.

Metrics:
  - WER / CER     : faster-whisper (large-v3, fp16 GPU) vs the manifest `text`, via jiwer
  - spk_sim       : campplus cosine(candidate 24k->16k, its reference wav) — same fbank
                    pipeline as cosyvoice/cli/frontend.py:_extract_spk_embedding
  - utmos         : torch.hub tarepan/SpeechMOS utmos22_strong (optional; --no-utmos to skip)
  - mel_l1_vs_gold: matcha mel (BLS partial: n_fft1920/80/24k/hop480/win1920/center=False)
                    L1 vs the gold wav for the same (reference_name, uid) — only meaningful
                    under token-replay (identical LLM tokens). Skipped if no --gold given.
  - artifacts     : lead_ms, n_dips, n_clicks, peak, clip_frac, rms, dur_s

Usage:
  python3 eval_quality.py --manifest english_basket_vanilla/samples_vanilla.json \
      --audio-root english_basket_vanilla --ref-dir reference_samples \
      --model-dir runtime/triton_trtllm/Fun-CosyVoice3-0.5B-2512 \
      --out scores_vanilla.json
  # optional gold for mel-L1:
      --gold-manifest <m.json> --gold-root <dir>
"""
import argparse, json, os, glob, sys, warnings
from functools import partial
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
warnings.filterwarnings("ignore")

CAMPPLUS_SR = 16000


# ---------- audio loading ----------
def load_wav(path, target_sr):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    t = torch.from_numpy(w).unsqueeze(0)
    if sr != target_sr:
        t = torchaudio.transforms.Resample(sr, target_sr)(t)
    return t  # [1, T]


# ---------- speaker embedding (campplus) — mirrors frontend._extract_spk_embedding ----------
class SpkSim:
    def __init__(self, campplus_onnx):
        opt = onnxruntime.SessionOptions()
        opt.intra_op_num_threads = 2
        opt.inter_op_num_threads = 2
        self.sess = onnxruntime.InferenceSession(
            campplus_onnx, sess_options=opt, providers=["CPUExecutionProvider"])
        self._cache = {}

    def embed(self, wav16):  # wav16: [1, T] @16k
        feat = kaldi.fbank(wav16, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        emb = self.sess.run(None, {self.sess.get_inputs()[0].name:
                                   feat.unsqueeze(0).cpu().numpy()})[0].flatten()
        return emb / (np.linalg.norm(emb) + 1e-9)

    def ref_embed(self, ref_wav_path):
        if ref_wav_path not in self._cache:
            self._cache[ref_wav_path] = self.embed(load_wav(ref_wav_path, CAMPPLUS_SR))
        return self._cache[ref_wav_path]

    def cosine(self, cand_wav_path, ref_wav_path):
        c = self.embed(load_wav(cand_wav_path, CAMPPLUS_SR))
        r = self.ref_embed(ref_wav_path)
        return float(np.dot(c, r))


# ---------- mel (matcha, BLS params) ----------
def get_mel_fn():
    from matcha.utils.audio import mel_spectrogram
    return partial(mel_spectrogram, n_fft=1920, num_mels=80, sampling_rate=24000,
                   hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)


def mel_of(path, mel_fn):
    w = load_wav(path, 24000)  # [1, T]
    return mel_fn(w).squeeze(0).cpu().numpy()  # [80, frames]


def mel_l1(cand_path, gold_path, mel_fn):
    a = mel_of(cand_path, mel_fn)
    b = mel_of(gold_path, mel_fn)
    n = min(a.shape[1], b.shape[1])
    if n == 0:
        return None
    return float(np.abs(a[:, :n] - b[:, :n]).mean())


# ---------- artifacts (reuse logic from analyze_breaks/basket_vanilla_pser) ----------
def artifacts(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    win = 480
    n = len(w) // win
    out = {"dur_s": len(w) / sr, "peak": float(np.abs(w).max()) if len(w) else 0.0,
           "rms": float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0,
           "clip_frac": float(np.mean(np.abs(w) > 0.99)) if len(w) else 0.0,
           "lead_ms": 0.0, "n_dips": 0, "n_clicks": 0}
    if n < 4:
        return out
    env = np.sqrt(np.array([np.mean(w[i*win:(i+1)*win]**2) for i in range(n)]) + 1e-12)
    lead = next((i for i, e in enumerate(env) if e > 0.02), 0)
    out["lead_ms"] = lead * win / sr * 1000
    dips = []
    for i in range(lead + 2, n - 1):
        if env[i] < 0.02:
            l = env[max(0, i-6):i]; r = env[i+1:i+7]
            if l.size and r.size and l.max() > 0.06 and r.max() > 0.06:
                if not dips or i - dips[-1] > 3:
                    dips.append(i)
    out["n_dips"] = len(dips)
    d = np.abs(np.diff(w[lead*win:]))
    idx = np.where(d > 0.3)[0]; clk = 0
    for i in idx:
        loc = d[max(0, i-3):i+4]
        if d[i] == loc.max() and d[i] > 3 * np.median(loc):
            clk += 1
    out["n_clicks"] = clk
    return out


# ---------- text normalization for WER ----------
def norm_text(s):
    import re
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def ref_from_system(system):
    for p in ("cosy3_", "vanilla_", "neutral2_"):
        if system.startswith(p):
            return system[len(p):]
    return system


def resolve(root, audio_path):
    """audio_path is relative to root (may include a subdir like ./cosy3_x/0001.wav)."""
    return os.path.normpath(os.path.join(root, audio_path.lstrip("./")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--ref-dir", default="reference_samples")
    ap.add_argument("--model-dir", default="runtime/triton_trtllm/Fun-CosyVoice3-0.5B-2512")
    ap.add_argument("--gold-manifest", default=None)
    ap.add_argument("--gold-root", default=None)
    ap.add_argument("--ref-name", default=None,
                    help="override reference_name for ALL entries (single-ref runs)")
    ap.add_argument("--limit", type=int, default=0, help="only first N entries (debug)")
    ap.add_argument("--no-utmos", action="store_true")
    ap.add_argument("--no-wer", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    if args.limit:
        man = man[:args.limit]
    print(f"[eval] {len(man)} entries from {args.manifest}", flush=True)

    # gold index for mel-L1
    gold_idx = {}
    if args.gold_manifest and args.gold_root:
        for g in json.load(open(args.gold_manifest)):
            rn = args.ref_name or ref_from_system(g.get("system", ""))
            gold_idx[(rn, g["uid"])] = resolve(args.gold_root, g["audio_path"])

    spk = SpkSim(os.path.join(args.model_dir, "campplus.onnx"))
    mel_fn = get_mel_fn() if gold_idx else None

    # WER model
    asr = None
    if not args.no_wer:
        from faster_whisper import WhisperModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ct = "float16" if dev == "cuda" else "int8"
        print(f"[eval] loading faster-whisper large-v3 ({dev}/{ct})...", flush=True)
        asr = WhisperModel("large-v3", device=dev, compute_type=ct)

    # UTMOS
    utmos = None
    if not args.no_utmos:
        try:
            utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
            utmos = utmos.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            print("[eval] UTMOS loaded", flush=True)
        except Exception as e:
            print(f"[eval] UTMOS unavailable ({e}); skipping", flush=True)
            utmos = None

    import jiwer
    per_file, refs_all, hyps_all = [], [], []
    for i, m in enumerate(man):
        path = resolve(args.audio_root, m["audio_path"])
        if not os.path.isfile(path):
            continue
        rn = args.ref_name or ref_from_system(m.get("system", ""))
        ref_wav = os.path.join(args.ref_dir, f"{rn}_16k.wav")
        rec = {"uid": m["uid"], "system": m.get("system", ""), "reference": rn}
        rec.update(artifacts(path))
        # spk-sim
        try:
            rec["spk_sim"] = spk.cosine(path, ref_wav) if os.path.isfile(ref_wav) else None
        except Exception as e:
            rec["spk_sim"] = None
        # WER
        if asr is not None:
            try:
                segs, _ = asr.transcribe(path, language="en", beam_size=5)
                hyp = norm_text(" ".join(s.text for s in segs))
                ref = norm_text(m["text"])
                rec["wer"] = float(jiwer.wer(ref, hyp)) if ref else None
                rec["cer"] = float(jiwer.cer(ref, hyp)) if ref else None
                rec["hyp"] = hyp
                if ref:
                    refs_all.append(ref); hyps_all.append(hyp)
            except Exception as e:
                rec["wer"] = rec["cer"] = None
        # UTMOS
        if utmos is not None:
            try:
                w16 = load_wav(path, 16000).to(next(utmos.parameters()).device)
                with torch.no_grad():
                    rec["utmos"] = float(utmos(w16, 16000).item())
            except Exception:
                rec["utmos"] = None
        # mel-L1 vs gold
        if mel_fn is not None:
            gp = gold_idx.get((rn, m["uid"]))
            rec["mel_l1"] = mel_l1(path, gp, mel_fn) if gp and os.path.isfile(gp) else None
        per_file.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(man)}", flush=True)

    def agg(key, pct=None):
        vals = [r[key] for r in per_file if r.get(key) is not None]
        if not vals:
            return None
        if pct is not None:
            return float(np.percentile(vals, pct))
        return float(np.mean(vals))

    aggregate = {
        "n": len(per_file),
        "wer_corpus": float(jiwer.wer(refs_all, hyps_all)) if refs_all else None,
        "cer_corpus": float(jiwer.cer(refs_all, hyps_all)) if refs_all else None,
        "spk_sim_mean": agg("spk_sim"), "spk_sim_p05": agg("spk_sim", 5),
        "utmos_mean": agg("utmos"), "utmos_p05": agg("utmos", 5),
        "mel_l1_mean": agg("mel_l1"), "mel_l1_p95": agg("mel_l1", 95),
        "lead_ms_mean": agg("lead_ms"),
        "n_dips_total": int(sum(r["n_dips"] for r in per_file)),
        "clicks_total": int(sum(r["n_clicks"] for r in per_file)),
        "files_with_clicks": int(sum(1 for r in per_file if r["n_clicks"] > 0)),
        "peak_max": max((r["peak"] for r in per_file), default=0.0),
        "runaway_count": int(sum(1 for r in per_file if r["dur_s"] > 18)),
    }
    json.dump({"manifest": args.manifest, "per_file": per_file, "aggregate": aggregate},
              open(args.out, "w"), indent=2)
    print(f"\n[eval] WROTE {args.out}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
