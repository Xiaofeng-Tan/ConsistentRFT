GPU_NUM=7 # 2,4,8
MODEL_PATH="./pretrained_ckpt/flux"
OUTPUT_DIR="offline_dataset/rl_embeddings"
IMG_DIR="offline_dataset/train"
torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding_offline.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --json_path "./offline_dataset/train.json" \
    --number_pair 20000 \
    --img_dir $IMG_DIR \
    # --height 720 \
    # --width 720 \
