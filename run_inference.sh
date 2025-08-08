#!/bin/bash

MULTI_GPU=${1:-0}
FP8=${2:-0}

export TOKENIZERS_PARALLELISM=false

export NPROC_PER_NODE=8
export ULYSSES_DEGREE=8
export RING_DEGREE=1

export PWD=$( realpath "$0"  )
echo $PWD
export PATH_TO=/beiw/HunyuanVideo_beiw/ckpts/hunyuan-video-t2v-720p/transformers
export DIT_CKPT_PATH=${PATH_TO}/mp_rank_00_model_states_fp8.pt

#PROFILE_PATH=/beiw/HunyuanVideo/profiling_output
PROFILE_PATH=/beiw/HunyuanVideo_bei
ATTN=fa
PROFILE_FILE=hunyuanvideo_inference_${ATTN}_b200

if [ $MULTI_GPU -eq 0 ]; then
    if [ $FP8 -eq 0 ]; then
	nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o ${PROFILE_PATH}/${PROFILE_FILE} python3 sample_video.py --video-size 720 1280 --video-length 129 --infer-steps 5 --prompt "A cat walks on the grass, realistic style." --flow-reverse --save-path ./results #--vae_trt
	
    else
	nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o $PROFILE_PATH/${PROFILE_FILE}_fp8 python3 sample_video.py --dit-weight ${DIT_CKPT_PATH} --video-size 720 1280 --video-length 129 --infer-steps 5 --prompt "A cat walks on the grass, realistic style." --flow-reverse --use-cpu-offload --use-fp8 --save-path ./results
	
    fi
else
    if [ $FP8 -eq 0 ]; then
	nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o $PROFILE_PATH/${PROFILE_FILE}_u${ULYSSES_DEGREE}r${RING_DEGREE} \
        torchrun --nproc_per_node=$NPROC_PER_NODE sample_video.py --video-size 720 1280 --video-length 129 --infer-steps 5 --prompt "A cat walks on the grass, realistic style." --flow-reverse --seed 42 --ulysses-degree $ULYSSES_DEGREE --ring-degree $RING_DEGREE --save-path ./results #--vae_trt
    else
	nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o $PROFILE_PATH/${PROFILE_FILE}_u${ULYSSES_DEGREE}r${RING_DEGREE}_fp8 torchrun --nproc_per_node=$NPROC_PER_NODE sample_video.py --dit-weight ${DIT_CKPT_PATH} --video-size 720 1280 --video-length 129 --infer-steps 5 --prompt "A cat walks on the grass, realistic style." --flow-reverse --seed 42 --ulysses-degree $ULYSSES_DEGREE --ring-degree $RING_DEGREE --use-fp8 --save-path ./results	
    fi
fi


# trt vae decoder testing
#nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o vae_trt python -m hyvideo.vae.autoencoder_test
