#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Usage: me.sh  pretrained_models_dir"
    exit
fi

PYTHON=${PYTHON:-"python"}

GPUS=0,1,2,3,4,5,6,7
NUM_GPUS=8

PORT=${PORT:-29600}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pretrained_models_dir=$1

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    $DIR/../tools_ivmcl/eval_ImageNet_X.py $pretrained_models_dir \
    --launcher pytorch --gpus $NUM_GPUS
