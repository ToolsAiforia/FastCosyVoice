import triton_python_backend_utils as pb_utils
from numpy.typing import NDArray
import numpy as np


def forward_audio_tokenizer(
    wav: NDArray[np.float32], wav_len: NDArray[np.int32],
) -> NDArray[np.int32]:
    wav_tensor = pb_utils.Tensor("reference_wav", wav)
    wav_len_tensor = pb_utils.Tensor("reference_wav_len", wav_len)

    inference_request = pb_utils.InferenceRequest(
        model_name="audio_tokenizer",
        requested_output_names=["prompt_speech_tokens"],
        inputs=[wav_tensor, wav_len_tensor],
        preferred_memory=pb_utils.PreferredMemory(
            pb_utils.TRITONSERVER_MEMORY_CPU,
        ),
    )

    inference_response = inference_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())

    prompt_speech_tokens = pb_utils.get_output_tensor_by_name(
        inference_response, "prompt_speech_tokens",
    )

    return prompt_speech_tokens.as_numpy()
