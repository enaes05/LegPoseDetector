set -x

CONFIG=$1
CKPT=$2
VIDEO=$3
OUTDIR=${4:-"./examples/res"}

python3 scripts/test.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video
