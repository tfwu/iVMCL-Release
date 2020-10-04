#!/usr/bin/env bash

if [ "$#" -lt 1 ]; then
    echo "Usage: me.sh relative_config_filename"
    exit
fi

PYTHON=${PYTHON:-"python"}

GPUS=0,1,2,3,4,5,6,7
NUM_GPUS=8

PORT=${PORT:-29600}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CONFIG_FILE=$1
CONFIG_FILENAME="$(cut -d'/' -f3 <<<$CONFIG_FILE)"
# CONFIG_FILENAME="$(echo '$CONFIG_FILE' | sed 's/.*\///')"
CONFIG_BASE="${CONFIG_FILENAME%.*}"
WORK_DIR=$DIR/../work_dirs/detection/$CONFIG_BASE
if [ -d $WORK_DIR ]; then
  if [ "$#" -gt 1 ]; then
    rm -r $WORK_DIR
    mkdir -p $WORK_DIR
  else
    echo "$WORK_DIR --- Already exists, delete it first if retraining is needed"
    exit
  fi
else
    mkdir -p $WORK_DIR
fi


OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPUS $PYTHON -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    $DIR/../tools/train.py $CONFIG_FILE \
    --launcher pytorch \
    --work-dir $WORK_DIR
