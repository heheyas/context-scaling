export MS_CACHE_HOME="<LOCAL_ROOT>/modelscope"
export MODELSCOPE_CACHE="<LOCAL_ROOT>/modelscope"

IMAGE_ROOT_PATH=$1
RESOLUTION=$2
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-8}
PORT=${PORT:-29500}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MULTI_GPU_FLAG=""
if [ "$PROCESSES" -gt 1 ]; then
  MULTI_GPU_FLAG="--multi_gpu"
fi

accelerate launch --num_machines 1 --num_processes $PROCESSES $MULTI_GPU_FLAG --mixed_precision "fp16" --main_process_port $PORT \
  $SCRIPT_DIR/compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug \
  --csv $SCRIPT_DIR/dpg_bench.csv
