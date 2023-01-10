mkdir log

CUDA_VISIBLE_DEVICES=1 python main.py --local_steps 4 --global_rounds 100 --batch_size 8 --dataset mnist --num_classes 10 \
    --num_clients 100 --join_ratio "0.1" --local_learning_rate 0.005 \
    --idx_path "../dataset_idx/mnist/dirichlet/dir_1_sparse/100client"  \
    --data_path "../data" \
    --log_path "log"