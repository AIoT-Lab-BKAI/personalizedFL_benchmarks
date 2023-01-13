#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=36:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Hung_perFL/logs/mnist/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load openmpi/4.1.3
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
source ~/venv/pytorch1.11+horovod/bin/activate

LOG_DIR="/home/aaa10078nj/Federated_Learning/Hung_perFL/logs/mnist/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

#Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r ./easyFL/benchmark/mnist/data ${DATA_DIR}

ALG="FedALA"
MODEL="cnn"
WANDB=1
ROUND=300
EPOCHS=16
BATCH=16
PROPOTION=0.25
GPU_ID=0
TASK="mnist_dir_1_sparse_N20_K5_E16"
IDX_DIR="../dataset_idx/mnist/dirichlet/dir_1_sparse/20client"
DATASET="mnist"
NUMCLASS=10
NUMCLIENT=20
LR=0.005

cd personalizedFL_benchmarks/FedALA

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --local_epochs ${EPOCHS} --global_rounds ${ROUND} --batch_size ${BATCH} --dataset ${DATASET} --num_classes ${NUMCLASS} --num_clients ${NUMCLIENT} --join_ratio ${PROPOTION} --local_learning_rate ${LR} --task ${TASK} --idx_path ${IDX_DIR}  --data_path ${DATA_DIR} --log_path ${LOG_DIR} 