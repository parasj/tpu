#!/bin/bash

#  Create Cloud TPU:
#  gcloud compute tpus execution-groups create \                                                                                                                                                                                                               ✘ 130
#   --tpu-only \
#   --name=efficientnet-tutorial \
#   --zone=us-central1-f \
#   --disk-size=300 \
#   --machine-type=n1-standard-8 \
#   --tf-version=1.15.5 \
#   --accelerator-type=v2-8
if [[ -z "${RUN_ID}" ]]; then
  read -p "Run ID: " RUN_ID
fi

if [[ -z "${MODEL_NAME}" ]]; then
  export MODEL_NAME="efficientnet-ld-b0"
  read -p "Mode name [${MODEL_NAME}]: " MODEL_NAME
fi

if [[ -z "${TPU_NAME}" ]]; then
  export TPU_NAME="efficientnet-tutorial"
  read -p "TPU name [${TPU_NAME}]: " TPU_NAME
fi

export PROJECT_ID=bair-common
export STORAGE_BUCKET=gs://imagenet-uscentral1f
export MODEL_DIR="${STORAGE_BUCKET}/tensorboard/${MODEL_NAME}_${RUN_ID}"
export LOG_PATH="/home/paras/tpu/logs/${MODEL_NAME}_${RUN_ID}.log"
export DATA_DIR=${STORAGE_BUCKET}/imagenet_processed
export PYTHONPATH=$PYTHONPATH:~/tpu/models

echo "Model name is $MODEL_NAME"
echo "Saving output to $MODEL_DIR"
echo "Logging to $LOG_PATH"
echo "Using $TPU_NAME"
cd ~/tpu/models/official/efficientnet

set -ex
python3 main.py \
  --tpu=${TPU_NAME} \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --model_name="${MODEL_NAME}" \
  --skip_host_call=false \
  --train_batch_size=1024 \
  --train_steps=1751592 |& tee "$LOG_PATH"
