# #!/bin/bash
# #$ -cwd
# #$ -l rt_G.small=1
# #$ -l h_rt=36:00:00
# #$ -o /home/aaa10078nj/Federated_Learning/Hung_perFL/logs/cifar10/$JOB_NAME_$JOB_ID.log
# #$ -j y

# source /etc/profile.d/modules.sh
# module load gcc/11.2.0
# module load openmpi/4.1.3
# module load cuda/11.5/11.5.2
# module load cudnn/8.3/8.3.3
# module load nccl/2.11/2.11.4-1
# module load python/3.10/3.10.4
# source ~/venv/pytorch1.11+horovod/bin/activate

# LOG_DIR="/home/aaa10078nj/Federated_Learning/Hung_perFL/logs/cifar10/$JOB_NAME_$JOB_ID"
# rm -r ${LOG_DIR}
# mkdir ${LOG_DIR}

# #Dataset
# DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
# cp -r ./easyFL/benchmark/cifar10/data ${DATA_DIR}

ALG="FedALA"
MODEL="cnn"
WANDB=1
ROUND=4000
EPOCHS=1
BATCH=16
PROPOTION=0.05
GPU_ID=0
TASK="cifar10_dir_1_sparse_N200_K10_E1"
IDX_DIR="../cifar10/dir_1_sparse/200client"

DATASET="./log"
NUMCLASS=./data
NUMCLIENT=cifar10
LR=10

cd personalizedFL_benchmarks/FedALA

CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --local_epochs ${EPOCHS} --global_rounds ${ROUND} --batch_size ${BATCH} --dataset ${DATASET} --num_classes ${NUMCLASS} --num_clients ${NUMCLIENT} --join_ratio ${PROPOTION} --local_learning_rate ${LR} --task ${TASK} --idx_path ${IDX_DIR}  --data_path ${DATA_DIR} --log_path ${LOG_DIR} 