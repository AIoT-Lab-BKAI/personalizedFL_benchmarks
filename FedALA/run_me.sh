mkdir log

CUDA_VISIBLE_DEVICES=1 python main.py --local_epochs 4 --global_rounds 5 --batch_size 8 --dataset mnist --num_classes 10 \
    --num_clients 100 --join_ratio "0.1" --local_learning_rate 0.005 \
    --task "mnist_dir1sparse_N100_K10" \
    --idx_path "../dataset_idx/mnist/dirichlet/dir_1_sparse/100client"  \
    --data_path "../../myPFL/benchmark/mnist/data" \
    --log_path "log" --wandb 0