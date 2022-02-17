#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=develbooster
#SBATCH --job-name=open_clip_debugging

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
srun python -u src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/p/scratch/ccstdl/shankar/CC3M/train/{00000..03318}.tar" \
    --warmup 10000 \
    --batch-size=64 \
    --lr=5e-4 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model RN50 \
    --dist-url="env://" \
    --local-loss \
    --ssl_scale 0.0  \
    --SLIP
