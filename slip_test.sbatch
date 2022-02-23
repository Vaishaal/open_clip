#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --partition=booster
#SBATCH --job-name=cc3m_slip_test_8_node

# load low-level libraries
ml purge
eval "$(/p/project/ccstdl/gordon2/miniconda3/bin/conda shell.bash hook)" # init conda
conda activate open_clip
ml use $OTHERSTAGES
ml GCC/10.3.0
ml NCCL/2.10.3-1-CUDA-11.3
ml OpenMPI
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=15980
export WORLD_SIZE=$SLURM_NTASKS

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR

cd /p/scratch/ccstdl/shankar/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export TIME=$(date +%s)
export JOB_NAME="SLIP_TEST_1_NODE_$TIME"
srun python -u src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/p/scratch/ccstdl/shankar/CC3M/train/{00000..03318}.tar" \
    --warmup 10000 \
    --batch-size=128\
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=10 \
    --workers=8 \
    --model ViT-B/16 \
    --dist-url="env://" \
    --gather-with-grad \
    --local-loss \
    --ssl_scale 1.0  \
    --SLIP \
    --name=$JOB_NAME

srun python -u src/training/main.py \
    --zeroshot-frequency 1 \
    --val-data="/p/scratch/ccstdl/gordon2/CC3M/validation/{00000..00015}.tar" \
    --imagenet-val="/p/scratch/ccstdl/gordon2/imagenet_val" \
    --model ViT-B/16 \
    --resume "/p/scratch/ccstdl/shankar/open_clip/logs/$JOB_NAME/checkpoints/epoch_10.pt" \
    --workers 8 \
    --local-loss \
    --SLIP \
    --name=$JOB_NAME

