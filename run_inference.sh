#!/bin/bash

MULTI_GPU=${1:-0}
FP8=${2:-0}

export TOKENIZERS_PARALLELISM=false

export NPROC_PER_NODE=8
export ULYSSES_DEGREE=8
export RING_DEGREE=1


SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`

export PATH_TO=${SCRIPTPATH}/ckpts/hunyuan-video-t2v-720p/transformers
export DIT_CKPT_PATH=${PATH_TO}/mp_rank_00_model_states_fp8.pt
export DIT_MODELOPT_CKPT_PATH=${PATH_TO}/mp_rank_00_model_states_modelopt_fp8.pt

PROFILE_PATH=${SCRIPTPATH}
ATTN=torch_cudnn #torch_cudnn
PROFILE_FILE=hunyuanvideo_inference_${ATTN}_b200

#export __LUNOWUD="-kgen:dump_source"
#export __LUNOWUD="-mlir:tile_aa:support_bmm2_quantize=on"

rm -rf ./onnx/model*
rm -rf ./engine/model*
rm -rf model*cache.lock

#rm -rf ${DIT_MODELOPT_CKPT_PATH}

if [ $MULTI_GPU -eq 0 ]; then
    if [ $FP8 -eq 0 ]; then
	nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o ${PROFILE_PATH}/${PROFILE_FILE} \
	     python3 sample_video.py --video-size 720 1280 --video-length 129 --infer-steps 5 --prompt "A cat walks on the grass, reaistic style." --flow-reverse --flow-shift 7.0 --embedded-cfg-scale 6.0 --seed 42 --save-path ./results #--vae-trt #--model-trt
    else
	nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o $PROFILE_PATH/${PROFILE_FILE}_fp8 \
	    python3 sample_video.py --dit-modelopt-weight ${DIT_MODELOPT_CKPT_PATH} --video-size 720 1280 --video-length 129 --infer-steps 5 --prompt "A cat walks on the grass, realistic style." --flow-reverse --flow-shift 7.0 --embedded-cfg-scale 6.0 --seed 42 --save-path ./results --vae-trt --use-modelopt-fp8 --model-trt #--use-modelopt-fp8
	
    fi
else
    if [ $FP8 -eq 0 ]; then
	#nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o $PROFILE_PATH/${PROFILE_FILE}_u${ULYSSES_DEGREE}r${RING_DEGREE} \
            torchrun --nproc_per_node=$NPROC_PER_NODE sample_video.py --video-size 720 1280 --video-length 129 --infer-steps 5 --prompt "A cat walks on the grass, realistic style." --flow-reverse --flow-shift 7.0 --embedded-cfg-scale 6.0 --seed 42 --ulysses-degree $ULYSSES_DEGREE --ring-degree $RING_DEGREE --save-path ./results --vae-trt --model-trt
	    
    else
	nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true -f true -o $PROFILE_PATH/${PROFILE_FILE}_u${ULYSSES_DEGREE}r${RING_DEGREE}_fp8 torchrun --nproc_per_node=$NPROC_PER_NODE sample_video.py --dit-weight ${DIT_CKPT_PATH} --video-size 720 1280 --video-length 129 --infer-steps 5 --prompt "A cat walks on the grass, realistic style." --flow-reverse --flow-shift 7.0 --embedded-cfg-scale 6.0 --seed 42 --ulysses-degree $ULYSSES_DEGREE --ring-degree $RING_DEGREE --use-fp8 --save-path ./results
	
    fi
fi
