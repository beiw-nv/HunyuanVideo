#!/bin/bash

#srun -A coreai_devtech_all -N1 -J coreai_devtech_all-pytorch_cudnn_attetion.inteactive_test -p batch --container-image=nvcr.io/nvidia/pytorch:25.05-py3 --container-mounts=/lustre/fsw/coreai_devtech_all/beiw:/beiw -t 04:00:00 --pty bash

#source install.sh

#git clone https://github.com/beiw-nv/flash-attention flash-attention_beiw
cd /beiw/flash-attention_beiw

# this is needed to build cute lib
pip install nvidia-cutlass-dsl

rm -rf build dist flash_attn.egg-info

pip install . -v --no-build-isolation

export PYTHONPATH=$PWD:$PYTHONPATH

#git clone https://github.com/beiw-nv/long-context-attention.git long-context-attention_beiw
cd /beiw/long-context-attention_beiw

#python -m pip install ninja

#python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.8.2

pip install .

#torchrun --nproc_per_node=4 --master_port=1234 ./test/test_hybrid_attn.py --sp_ulysses_degree 2 --ring_impl_type 'basic_flashinfer' --attn_impl flashinfer

#torchrun --nproc_per_node=4 --master_port=1234 ./test/test_hybrid_attn.py --sp_ulysses_degree 2 --ring_impl_type 'basic_torch' --attn_impl torch

#torchrun --nproc_per_node=4 --master_port=1234 ./test/test_hybrid_attn.py --sp_ulysses_degree 2 --ring_impl_type 'basic' --attn_impl fa_cute

#git clone https://github.com/beiw-nv/xDiT xDiT_beiw
cd /beiw/xDiT_beiw

pip install -e .

cd /beiw/HunyuanVideo_beiw

python -m pip install -r requirements.txt

#bash run_inference.sh 1 0
