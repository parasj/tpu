SSH into instance:
`ssh paras@35.232.169.150`

CD to experiment directory:
`cd ~/tpu`

Run script with TPUs (increment run ID each time):
`RUN_ID=0011 MODEL_NAME='efficientnet-b0' TPU_NAME='efficientnet-tutorial' bash run.sh`

Valid experiment configurations:
* `MODEL_NAME='efficientnet-ld-b0' TPU_NAME='efficientnet-tutorial-3' bash run.sh`
* `MODEL_NAME='efficientnet-ld-b0-d2d6d7d9d12d13' TPU_NAME='efficientnet-tutorial-1' bash run.sh`
* `MODEL_NAME='efficientnet-ld-b0-d2d6d9d12' TPU_NAME='efficientnet-tutorial-2' bash run.sh`
