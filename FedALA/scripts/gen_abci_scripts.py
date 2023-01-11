import os
from pathlib import Path


dataset = "cifar10"
noniid = "dir_1_sparse"
N = 100
K = 10
total_epochs = 4000
batch_size = 16

model = "cnn"
algos = ["FedALA"]
data_folder = f"./data"
log_folder = f"./log"

if not Path(f"./{dataset}/{N}_clients").exists():
    os.makedirs(f"./{dataset}/{N}_clients")


header_text = "\
# #!/bin/bash\n\
# #$ -cwd\n\
# #$ -l rt_G.small=1\n\
# #$ -l h_rt=36:00:00\n\
# #$ -o /home/aaa10078nj/Federated_Learning/Hung_test/logs/motiv1/$JOB_NAME_$JOB_ID.log\n\
# #$ -j y\n\n\
# source /etc/profile.d/modules.sh\n\
# module load gcc/11.2.0\n\
# module load openmpi/4.1.3\n\
# module load cuda/11.5/11.5.2\n\
# module load cudnn/8.3/8.3.3\n\
# module load nccl/2.11/2.11.4-1\n\
# module load python/3.10/3.10.4\n\
# source ~/venv/pytorch1.11+horovod/bin/activate\n\n\
# LOG_DIR=\"/home/aaa10078nj/Federated_Learning/Hung_test/logs/motiv1/cifar10/$JOB_NAME_$JOB_ID\"\n\
# rm -r ${LOG_DIR}\n\
# mkdir ${LOG_DIR}\n\n\
# #Dataset\n\
# DATA_DIR=\"$SGE_LOCALDIR/$JOB_ID/\"\n\
# cp -r ./easyFL/benchmark/cifar10/data ${DATA_DIR}\n\n\
"

formated_command = "\
ALG=\"{}\"\n\
MODEL=\"{}\"\n\
WANDB=1\n\
ROUND={}\n\
EPOCHS={}\n\
BATCH={}\n\
PROPOTION={:>.2f}\n\
GPU_ID=0\n\
TASK=\"{}\"\n\
IDX_DIR=\"../{}/{}/{}client\"\n\n\
DATASET=\"{}\"\n\
NUMCLASS={}\n\
NUMCLIENT={}\n\
LR={}\n\
\ncd FedALA\n\n\
"
body_text = "\
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --local_epochs ${EPOCHS} --global_rounds ${ROUND} \
--batch_size ${BATCH} --dataset ${DATASET} --num_classes ${NUMCLASS} \
--num_clients ${NUMCLIENT} --join_ratio ${PROPOTION} --local_learning_rate ${LR} \
--task ${TASK} \
--idx_path ${IDX_DIR}  \
--data_path ${DATA_DIR} \
--log_path ${LOG_DIR} \
"
        
for local_epochs in [1, 8, 16, 32]:
    task_name = f"{dataset}_{noniid}_N{N}_K{K}_E{local_epochs}"

    for algo in algos:
        command = formated_command.format(
            algo, model, int(total_epochs/local_epochs), local_epochs, batch_size, K/N, 
            task_name, dataset, noniid, N, log_folder, data_folder, 
            dataset, 10, N, 0.005
        )

        file = open(f"./{dataset}/{N}_clients/{task_name}_{algo}.sh", "w")
        file.write(header_text + command + body_text)
        file.close()