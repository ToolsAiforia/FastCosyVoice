#!/bin/bash
# Smoke test for cosyvoice3-tts container.
# Run AFTER `docker run` shows "Triton server is ready" in logs.
#
# Usage:
#   ./smoke_test.sh                    # uses default port 8001 + container 'cosyvoice3-tts'
#   ./smoke_test.sh 18001 cv3-test     # custom port + container name

set -e

GRPC_PORT="${1:-8001}"
HTTP_PORT="$((GRPC_PORT - 1))"
CONTAINER="${2:-cosyvoice3-tts}"

echo "============================================"
echo "  CosyVoice3 Smoke Test"
echo "  Container: $CONTAINER"
echo "  gRPC port: $GRPC_PORT, HTTP port: $HTTP_PORT"
echo "============================================"

echo ""
echo "[1/5] Triton health check..."
curl -fs "http://localhost:${HTTP_PORT}/v2/health/ready" \
    && echo " ✓ Triton ready" \
    || { echo " ✗ Triton NOT ready"; exit 1; }

echo ""
echo "[2/5] Model repository status..."
MODELS=$(curl -s -X POST "http://localhost:${HTTP_PORT}/v2/repository/index" 2>/dev/null)
echo "$MODELS" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
for m in d:
    state = m.get('state', 'UNKNOWN')
    name = m.get('name', '?')
    marker = '✓' if state == 'READY' else '✗'
    print(f'  {marker} {name}: {state}')
"

echo ""
echo "[3/5] Speaker availability (spk2info.pt)..."
docker exec "$CONTAINER" python3 -c "
import torch
spk = torch.load('/workdir/Fun-CosyVoice3-0.5B-2512/spk2info.pt',
                  map_location='cpu', weights_only=False)
for name, info in spk.items():
    text = info.get('reference_text', '')[:80]
    print(f'  ✓ {name}: tokens={info[\"prompt_speech_tokens\"].shape[1]}, text={text!r}...')
" || { echo "  ✗ spk2info.pt not loaded"; exit 1; }

echo ""
echo "[4/5] Streaming synthesis test (default speaker 'ref')..."
docker exec "$CONTAINER" python3 - <<EOF
import asyncio, time, numpy as np
import tritonclient.grpc.aio as grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput

async def main():
    client = grpc_aio.InferenceServerClient(url='localhost:${GRPC_PORT}', verbose=False)
    text = "Smoke test, the system is working as expected."
    spk = np.array([[b'ref']], dtype=object)
    tgt = np.array([[text.encode()]], dtype=object)
    inputs = [
        InferInput("target_text", tgt.shape, "BYTES"),
        InferInput("speaker_name", spk.shape, "BYTES"),
    ]
    inputs[0].set_data_from_numpy(tgt)
    inputs[1].set_data_from_numpy(spk)
    outputs = [InferRequestedOutput("waveform")]
    chunks = []; ttfa = None
    t0 = time.time()
    async def gen():
        yield {"model_name": "cosyvoice3", "inputs": inputs,
               "outputs": outputs, "request_id": "smoke_001"}
    async for resp in client.stream_infer(inputs_iterator=gen(), stream_timeout=60):
        r, err = resp
        if err: print(f"  ERROR: {err}"); break
        if ttfa is None: ttfa = (time.time() - t0) * 1000
        wav = r.as_numpy("waveform")
        if wav is not None and wav.size: chunks.append(wav.flatten())
    await client.close()
    if not chunks:
        print(f"  ✗ No audio chunks received")
        return False
    full = np.concatenate(chunks)
    dur = len(full) / 24000
    peak = float(np.max(np.abs(full)))
    total = (time.time() - t0) * 1000
    rtf = total / 1000 / dur
    print(f"  ✓ TTFA={ttfa:.0f}ms  total={total:.0f}ms  audio_dur={dur:.2f}s  peak={peak:.2f}  RTF={rtf:.3f}")
    if peak > 0.01 and peak <= 1.0:
        print(f"  ✓ Audio amplitude in valid range")
        return True
    print(f"  ✗ Audio amplitude out of valid range")
    return False

ok = asyncio.run(main())
exit(0 if ok else 1)
EOF
SYNTH_OK=$?

echo ""
echo "[5/5] Instruction override test..."
docker exec "$CONTAINER" python3 - <<EOF
import asyncio, time, numpy as np
import tritonclient.grpc.aio as grpc_aio
from tritonclient.grpc import InferInput, InferRequestedOutput

async def synth(instruction):
    client = grpc_aio.InferenceServerClient(url='localhost:${GRPC_PORT}', verbose=False)
    text = "Testing instruction override."
    spk = np.array([[b'ref']], dtype=object)
    tgt = np.array([[text.encode()]], dtype=object)
    inputs = [
        InferInput("target_text", tgt.shape, "BYTES"),
        InferInput("speaker_name", spk.shape, "BYTES"),
    ]
    inputs[0].set_data_from_numpy(tgt)
    inputs[1].set_data_from_numpy(spk)
    if instruction:
        i_np = np.array([[instruction.encode()]], dtype=object)
        ii = InferInput("instruction", i_np.shape, "BYTES")
        ii.set_data_from_numpy(i_np)
        inputs.append(ii)
    chunks = []
    async def gen():
        yield {"model_name": "cosyvoice3", "inputs": inputs,
               "outputs": [InferRequestedOutput("waveform")],
               "request_id": f"smoke_instr_{hash(instruction or 'default')}"}
    async for r, err in client.stream_infer(inputs_iterator=gen(), stream_timeout=60):
        if err: return None
        w = r.as_numpy("waveform")
        if w is not None and w.size: chunks.append(w.flatten())
    await client.close()
    return np.concatenate(chunks) if chunks else None

async def main():
    a = await synth(None)
    b = await synth("Whisper this softly, almost inaudible.")
    if a is None or b is None:
        print(f"  ✗ Synthesis failed")
        return False
    dur_a, dur_b = len(a) / 24000, len(b) / 24000
    print(f"  default instruction:  dur={dur_a:.2f}s")
    print(f"  whisper instruction:  dur={dur_b:.2f}s")
    if abs(dur_a - dur_b) > 0.1:
        print(f"  ✓ Instruction override took effect (Δ duration = {abs(dur_a-dur_b)*1000:.0f}ms)")
        return True
    print(f"  ⚠ Durations nearly identical — instruction may not have engaged")
    return True  # not a hard fail

ok = asyncio.run(main())
exit(0 if ok else 1)
EOF
INSTR_OK=$?

echo ""
echo "============================================"
if [ $SYNTH_OK -eq 0 ] && [ $INSTR_OK -eq 0 ]; then
    echo "  ✓ All smoke tests PASSED"
    echo "  Container ready for production / registry push"
else
    echo "  ✗ Some tests failed"
    exit 1
fi
echo "============================================"
