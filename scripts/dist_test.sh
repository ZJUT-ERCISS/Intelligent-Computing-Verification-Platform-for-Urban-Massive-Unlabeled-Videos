export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

export PYTHONPATH='/home/shy/code_hik/video_model/'
MKL_SERVICE_FORCE_INTEL=1 \
# Arguments starting from the forth one are captured by ${@:4}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$MASTER_PORT \
    test.py $CONFIG -C $CHECKPOINT --launcher pytorch ${@:4}

# python -m torch.distributed.launch --nproc_per_node=1 --master_port=22077 test.py /home/shy/code_hik/video_model/config/ske_gcn/config_aagcn_vivit.py -C /home/shy/code_hik/video_model/.vscode/resources/pretrained_model.pth --launcher pytorch ${@:4}