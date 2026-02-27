import numpy as np
from numpy.typing import NDArray
import triton_python_backend_utils as pb_utils


def forward_speaker_embedding(wav: NDArray[np.float32]) -> NDArray[np.float16]:
    inference_request = pb_utils.InferenceRequest(
        model_name="speaker_embedding",
        requested_output_names=["prompt_spk_embedding"],
        inputs=[pb_utils.Tensor("reference_wav", wav)],
    )

    inference_response = inference_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())

    prompt_spk_embedding = pb_utils.get_output_tensor_by_name(
        inference_response,
        "prompt_spk_embedding",
    )
    return prompt_spk_embedding.as_numpy()
