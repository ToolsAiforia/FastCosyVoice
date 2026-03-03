# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os

import logging

import torch
from torch.utils.dlpack import to_dlpack
from torch.nn import functional as F

import triton_python_backend_utils as pb_utils

import numpy as np
from typing import NamedTuple
from token2wav_lib import CosyVoice2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ORIGINAL_VOCAB_SIZE = 151663
torch.set_num_threads(1)

class Prompt(NamedTuple):
    speech_tokens: NDArray[np.int32]
    speech_feat: NDArray[np.float16]
    spk_embedding: NDArray[np.float16]


class Inputs(NamedTUple):
    request_id: str
    target_speech_tokens: NDArray
    prompt: Prompt


class TritonPythonModel:

    def initialize(self, args):
        # Parse model parameters
        parameters = json.loads(args["model_config"])["parameters"]
        model_params = {key: value["string_value"] for key, value in parameters.items()}
        model_dir = model_params["model_dir"]

        # Initialize device and vocoder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing vocoder from {model_dir} on {self.device}")
        
        self.token2wav_model = CosyVoice2(
            model_dir, load_jit=False, load_trt=True, fp16=True, device=self.device
        )

    def execute(
        self,
        requests: list["pb_utils.InferenceRequest"],
    ) -> list["pb_utils.InferenceResponse"]:
        return [self._infer(request) for request in requests]

    def _infer(
        self,
        request: "pb_utils.InferenceRequest",
    ) -> "pb_utils.InferenceResponse":
        inputs = self._get_inputs(request)
        # Black magic we meet yet again
        target_speech_tokens = torch.from_numpy(target_speech_tokens_tensor).to(
            self.device
        )
        prompt_speech_tokens = torch.from_numpy(prompt_speech_tokens_tensor).to(
            self.device
        )
        prompt_speech_feat = torch.from_numpy(prompt_speech_feat_tensor).to(
            self.device
        )
        prompt_spk_embedding = torch.from_numpy(prompt_spk_embedding_tensor).to(
            self.device
        )
        prompt_speech_tokens = prompt_speech_tokens - ORIGINAL_VOCAB_SIZE
        
        audio_hat = self._stream_step(inputs) if inputs.token_offset else self._full_run(inputs)

        wav_tensor = pb_utils.Tensor("waveform", audio_hat)
        return pb_utils.InferenceResponse(output_tensors=[wav_tensor])
    
    def _stream_step(self, inputs: Inputs) -> NDArray[np.float32]:
        audio_hat = self.token2wav_model.model.token2wav(
            token=target_speech_tokens,
            prompt_token=prompt_speech_tokens,
            prompt_feat=prompt_speech_feat,
            embedding=prompt_spk_embedding,
            token_offset=token_offset,
            uuid=request_id,
            stream=not finalize,
            finalize=finalize,
        )
        # bad idea since finalize could never happen
        # we should use some cache utility instead
        if finalize:
            self.token2wav_model.model.hift_cache_dict.pop(request_id)
        return audio_hat.squeeze(0).cpu().numpy()

    def _full_run(self, inputs: Inputs) -> NDArray[np.float32]:
        tts_mel, _ = self.token2wav_model.model.flow.inference(
            token=target_speech_tokens,
            token_len=torch.tensor(
                [target_speech_tokens.shape[1]], dtype=torch.int32
            ).to(self.device),
            prompt_token=prompt_speech_tokens,
            prompt_token_len=torch.tensor(
                [prompt_speech_tokens.shape[1]], dtype=torch.int32
            ).to(self.device),
            prompt_feat=prompt_speech_feat,
            prompt_feat_len=torch.tensor(
                [prompt_speech_feat.shape[1]], dtype=torch.int32
            ).to(self.device),
            embedding=prompt_spk_embedding,
            streaming=False,
            finalize=True,
        )

        audio_hat, _ = self.token2wav_model.model.hift.inference(
            speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0)
        )
        return audio_hat.squeeze(0).cpu().numpy()
    
    def _get_inputs(self, request: "pb_utils.InferenceRequest") -> Inputs:
        request_id = request.request_id()
        target_speech_tokens_tensor = pb_utils.get_input_tensor_by_name(
            request, "target_speech_tokens"
        ).as_numpy()

        prompt_speech_tokens = pb_utils.get_input_tensor_by_name(
            request, "prompt_speech_tokens"
        ).as_numpy()
        prompt_speech_tokens = prompt_speech_tokens_tensor.as_numpy()
        prompt_speech_feat = pb_utils.get_input_tensor_by_name(
            request, "prompt_speech_feat"
        ).as_numpy()
        prompt_spk_embedding = pb_utils.get_input_tensor_by_name(
            request, "prompt_spk_embedding"
        ).as_numpy()

        # We set token_offset as an optional input to support streaming/offline tts. It has to be None when offline tts.
        token_offset_tensor = pb_utils.get_input_tensor_by_name(request, "token_offset")
        finalize_tensor = pb_utils.get_input_tensor_by_name(request, "finalize")
        
        token_offset, finalize = None, None
        if token_offset_tensor is not None and finalize_tensor is not None:
            token_offset = token_offset_tensor.as_numpy()
            finalize = finalize_tensor.as_numpy().item()

        return Inputs(
            request_id=request_id,
            target_speech_tokens=target_speech_tokens,
            token_offset=token_offset,
            finalize=finalize,
            prompt=Prompt(
                speech_tokens=prompt_speech_tokens,
                speech_feat=prompt_speech_feat,
                spk_embedding=prompt_spk_embedding,
            ),
        )
