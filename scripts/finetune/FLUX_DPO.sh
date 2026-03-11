export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online


image_path="images_dpo_mix"
mkdir -p $image_path
echo $image_path

#sudo apt-get update
#yes | sudo apt-get install python3-tk

# git clone https://gitee.com/ju_siyuan/HPSv2.git
# cd HPSv2
# pip install -e . 
# cd ..

#pip3 install trl
torchrun --nproc_per_node=8 --master_port 19007 \
    fastvideo/train_DPO_flux.py \
    --seed 42 \
    --pretrained_model_name_or_path ./pretrained_ckpt/flux/ \
    --cache_dir ~/.cache/ \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 8 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 8 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 301 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 20 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_dpo_same_clip \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 16 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --use_hpsv2 \
    --num_generations 2 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --skip_rate 1.0 \
    --skip_rate_range 1.0 \
    --save_image_path $image_path \
    --dpo_beta 3000 \
    --lora_alpha 256 \
    --lora_rank 128 \
    --project "dpo" \
    --init_same_noise \
    # --mix \
    # --mix_rate 0.15 \
    # --mix_period 40 \
    # --consist \
    # --consist_weight 0.000001 \
    #--resume_from_lora_checkpoint ./data/outputs/grpo_dpo_same/lora-checkpoint-100-0 \
    #--init_same_noise \
    # --use_reference \
    # --diversity \
    # --diversity_step 12 \
    # --diversity_rate 8.0 \
    # --select_by "feature" \
    # --select_method "kmeans" \
    # --remain_div \
    #--resume_steps 40 \
    #--resume_from_checkpoint data/outputs/grpo_mix_8GPU_3/checkpoint-40-0 \
    #--skip_rate 0.7 \
    #--skip_rate_range 0.7 \
    #--resume_steps 140 \
    #--resume_from_checkpoint data/outputs/grpo_mix_8GPU/checkpoint-140-0 \
    #--skip_rate 0.5 \
    #--skip_rate_range 0.5 \
    #--use_ema \
    #--ema_decay 0.995 \
    #  --use_hpsv2_clip \
    #   --init_same_noise \