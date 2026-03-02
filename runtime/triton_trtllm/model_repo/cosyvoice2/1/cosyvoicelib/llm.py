from typing import Generator, Any, AsyncGenerator
from numpy.typing import NDArray
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class LLM:
    def __init__(self, llm_tokenizer_dir: str):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_dir)
        self.prompt_template = "<|sos|>{input_text}<|task_id|>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eos1|>")
        self.max_tokens = 750
        self.top_p = 0.95
        self.top_k = 50
        self.temperature = 0.8
        self.random_seed = 42
        self.repetition_penalty = 1.1

    def infer(self, input_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        llm_request = self._prepare_llm_request(input_ids)
        llm_response = llm_request.exec()
        return self._process_response(llm_response)

    def stream_infer(
        self,
        input_ids: NDArray[np.int32],
    ) -> Generator[NDArray[np.int32], None, None]:
        llm_request = self._prepare_llm_request(input_ids, streaming=True)
        llm_responses = llm_request.exec(decoupled=True)

        for llm_response in llm_responses:
            yield self._process_response(llm_response)

    def parse_input(
        self,
        text: str,
        prompt_text: str,
        prompt_speech_tokens: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        total_text = f"{prompt_text}{text}"
        prompt = self.prompt_template.format(input_text=total_text)
        input_ids = self.tokenizer.encode(prompt)
        input_ids = np.array(input_ids, dtype=np.int32)[None]
        input_ids = np.concatenate([input_ids, prompt_speech_tokens], axis=1)
        return input_ids

    def _prepare_llm_request(
        self,
        input_ids: NDArray,
        streaming: bool = False,
    ) -> "pb_utils.InferenceRequest":
        input_dict = {
            "request_output_len": np.array([[self.max_tokens]], dtype=np.int32),
            "end_id": np.array([[self.eos_token_id]], dtype=np.int32),
            "pad_id": np.array([[self.eos_token_id]], dtype=np.int32),
            "streaming": np.array([[streaming]], dtype=np.bool_),
            "runtime_top_p": np.array([[self.top_p]], dtype=np.float32),
            "runtime_top_k": np.array([[self.top_k]], dtype=np.int32),
            "temperature": np.array([[self.temperature]], dtype=np.float32),
            "repetition_penalty": np.array(
                [[self.repetition_penalty]], dtype=np.float32
            ),
            "random_seed": np.array([[self.random_seed]], dtype=np.uint64),
            "input_ids": input_ids,
            "input_lengths": np.array([[input_ids.shape[1]]], dtype=np.int32),
        }
        input_tensor_list = [pb_utils.Tensor(k, v) for k, v in input_dict.items()]
        return pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=input_tensor_list,
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU,
            ),
        )

    def _process_response(self, llm_response: "pb_utils.InferenceResponse") -> NDArray[np.int32]:
        if llm_response.has_error():
            return pb_utils.InferenceResponse(
                error=pb_utils.TritonError(
                    pb_utils.TritonModelException(llm_response.error().message()),
                    pb_utils.TritonError.CANCELLED,
                ),
            )

        output_ids = pb_utils.get_output_tensor_by_name(
            llm_response, "output_ids"
        ).as_numpy()
        seq_lens = pb_utils.get_output_tensor_by_name(
            llm_response, "sequence_length"
        ).as_numpy()

        return output_ids[0][0][: seq_lens[0][0]]
