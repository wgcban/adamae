# Set the path to save checkpoints
OUTPUT_DIR='experiments/adamae_K400finetune_pretraink400_videomae_base_patch16_224_learnable_masking_ratio_0.95/'
# path to K400 set (train.csv/val.csv/test.csv)
DATA_PATH='/data/wbandar1/datasets/ss2/list_ssv2/'
# path to pretrain model 
MODEL_PATH='experiments/kinetics400_videomae_pretrain_pretrain_videomae_base_patch16_224_learnable_masking_ratio_0.95/checkpoint-1239.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
    finetune_class.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 14 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 5 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval \
    --test_num_segment 7 \
    --test_num_crop 3

