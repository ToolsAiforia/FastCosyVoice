#!/bin/bash
# Copyright (c) 2025 NVIDIA (authors: Yuekai Zhang)
export CUDA_VISIBLE_DEVICES=0
cosyvoice_path=/workspace/CosyVoice
export PYTHONPATH=${cosyvoice_path}:$PYTHONPATH
export PYTHONPATH=${cosyvoice_path}/third_party/Matcha-TTS:$PYTHONPATH
stage=$1
stop_stage=$2

llm_dir=./Fun-CosyVoice3-0.5B/llm
model_dir=./Fun-CosyVoice3-0.5B

trt_dtype=bfloat16
trt_weights_dir=./trt_weights_${trt_dtype}
# trt_weights_dir=./Fun-CosyVoice3-0.5B/hf_merged
trt_engines_dir=./trt_engines_${trt_dtype}

model_repo=./model_repo_cosyvoice3

use_spk2info_cache=False

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading CosyVoice3-0.5B"
    huggingface-cli download --local-dir $model_dir FunAudioLLM/Fun-CosyVoice3-0.5B-2512
    # python3 scripts/
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Converting checkpoint to TensorRT weights"
    python3 scripts/convert_checkpoint.py \
       --model_dir $llm_dir \
       --output_dir $trt_weights_dir \
       --dtype $trt_dtype || exit 1

    echo "Building TensorRT engines"
    trtllm-build \
      --checkpoint_dir $trt_weights_dir \
      --output_dir $trt_engines_dir \
      --max_batch_size 16 \
      --max_num_tokens 32768 \
      --gemm_plugin $trt_dtype || exit 1

fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Creating model repository"
    rm -rf $model_repo
    mkdir -p $model_repo
    cosyvoice_dir="cosyvoice3"

    cp -r ./model_repo/${cosyvoice_dir} $model_repo
    cp -r ./model_repo/tensorrt_llm $model_repo
    cp -r ./model_repo/token2wav $model_repo
    if [ $use_spk2info_cache == "False" ]; then
        cp -r ./model_repo/audio_tokenizer $model_repo
        cp -r ./model_repo/speaker_embedding $model_repo
    fi

    ENGINE_PATH=$trt_engines_dir
    MAX_QUEUE_DELAY_MICROSECONDS=0
    MODEL_DIR=$model_dir
    LLM_TOKENIZER_DIR=$llm_dir
    BLS_INSTANCE_NUM=4
    TRITON_MAX_BATCH_SIZE=16
    DECOUPLED_MODE=True # True for streaming, False for offline

    python3 scripts/fill_template.py -i ${model_repo}/token2wav/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/${cosyvoice_dir}/config.pbtxt model_dir:${MODEL_DIR},bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32
    if [ $use_spk2info_cache == "False" ]; then
        python3 scripts/fill_template.py -i ${model_repo}/audio_tokenizer/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
        python3 scripts/fill_template.py -i ${model_repo}/speaker_embedding/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
   echo "Starting Triton server"
   tritonserver --model-repository $model_repo
fi
