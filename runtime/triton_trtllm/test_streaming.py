"""Quick streaming test for CosyVoice3 in decoupled mode.

Usage:
    python test_streaming.py              # use default speaker
    python test_streaming.py emily        # use speaker "emily" from spk2info.pt
"""
import asyncio
import sys
import time
import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as grpcclient_aio
from tritonclient.grpc import InferInput, InferRequestedOutput


async def main(speaker_name=None):
    url = "localhost:8001"
    model_name = "cosyvoice3"
    target_text = "Hello! This is a test of the streaming mode for CosyVoice three. The audio should arrive in multiple chunks."

    client = grpcclient_aio.InferenceServerClient(url=url, verbose=False)

    # Build inputs
    target_np = np.array([[target_text.encode("utf-8")]], dtype=object)
    inp_target = InferInput("target_text", target_np.shape, "BYTES")
    inp_target.set_data_from_numpy(target_np)
    out = InferRequestedOutput("waveform")

    inputs = [inp_target]
    if speaker_name:
        spk_np = np.array([[speaker_name.encode("utf-8")]], dtype=object)
        inp_spk = InferInput("speaker_name", spk_np.shape, "BYTES")
        inp_spk.set_data_from_numpy(spk_np)
        inputs.append(inp_spk)

    print(f"Sending streaming request: \"{target_text}\"")
    if speaker_name:
        print(f"Using speaker: {speaker_name}")
    start = time.time()

    # Use stream_infer for decoupled mode
    async def request_gen():
        yield {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": [out],
            "request_id": "stream_test_001",
        }

    chunks = []
    chunk_idx = 0
    first_chunk_time = None

    response_iterator = client.stream_infer(request_gen())
    async for reply, err in response_iterator:
        if err is not None:
            print(f"Error: {err}")
            break
        wav = reply.as_numpy("waveform").reshape(-1)
        elapsed = time.time() - start
        if chunk_idx == 0:
            first_chunk_time = elapsed
        duration = len(wav) / 24000
        print(f"  Chunk {chunk_idx}: {len(wav)} samples ({duration:.2f}s), elapsed={elapsed:.2f}s")
        chunks.append(wav)
        chunk_idx += 1

    total_time = time.time() - start

    if chunks:
        full_audio = np.concatenate(chunks)
        full_duration = len(full_audio) / 24000
        output_path = "/home/b.zhumash/FastCosyVoice/synth_results/streaming_test.wav"
        sf.write(output_path, full_audio, 24000)
        print(f"\nResults:")
        print(f"  Total chunks: {chunk_idx}")
        print(f"  First chunk latency: {first_chunk_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Audio duration: {full_duration:.2f}s")
        print(f"  RTF: {total_time / full_duration:.2f}")
        print(f"  Saved to: {output_path}")
    else:
        print("No audio chunks received!")

    await client.close()


if __name__ == "__main__":
    speaker = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(speaker_name=speaker))
