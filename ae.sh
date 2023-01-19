#!/bin/bash

export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.3
# sudo rm -r /usr/local/cuda
# sudo ln -s /usr/local/cuda-11.3  /usr/local/cuda
# source env_romp/bin/activate
conda activate hdgcn