import numpy as np
from numpy.typing import NDArray
import triton_python_backend_utils as pb_utils


def forward_token2wav(
    request_id: str,
    target_speech_tokens: NDArray[np.int32],
    prompt_speech_tokens: NDArray[np.int32],
    prompt_speech_feat: NDArray[np.float16],
    prompt_spk_embedding: NDArray[np.float16],
    token_offset: int | None = None,
    finalize: bool = False,
) -> NDArray[np.float32]:
    target_speech_tokens_tensor = pb_utils.Tensor(
        "target_speech_tokens", target_speech_tokens
    )
    prompt_speech_tokens_tensor = pb_utils.Tensor(
        "prompt_speech_tokens", prompt_speech_tokens
    )
    prompt_speech_feat_tensor = pb_utils.Tensor(
        "prompt_speech_feat", prompt_speech_feat
    )
    prompt_spk_embedding_tensor = pb_utils.Tensor(
        "prompt_spk_embedding", prompt_spk_embedding
    )

    inputs_tensor = [
        target_speech_tokens_tensor,
        prompt_speech_tokens_tensor,
        prompt_speech_feat_tensor,
        prompt_spk_embedding_tensor,
    ]

    if token_offset is not None:
        token_offset_tensor = pb_utils.Tensor(
            "token_offset", np.array([[token_offset]], dtype=np.int32)
        )
        finalize_tensor = pb_utils.Tensor(
            "finalize", np.array([[finalize]], dtype=np.bool_)
        )
        inputs_tensor.append(token_offset_tensor)
        inputs_tensor.append(finalize_tensor)

    inference_request = pb_utils.InferenceRequest(
        model_name="token2wav",
        requested_output_names=["waveform"],
        inputs=inputs_tensor,
        request_id=request_id,
        preferred_memory=pb_utils.PreferredMemory(
            pb_utils.TRITONSERVER_MEMORY_CPU,
        ),
    )

    inference_response = inference_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())

    waveform = pb_utils.get_output_tensor_by_name(inference_response, "waveform")

    return waveform.as_numpy()
