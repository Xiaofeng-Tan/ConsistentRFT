import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from safetensors.torch import load_file
import argparse
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import copy
from fastvideo.models.reward_model.image_reward import ImageRewardModel
from fastvideo.models.reward_model.pick_score import PickScoreRewardModel
from fastvideo.models.reward_model.unified_reward import UnifiedRewardModel
from fastvideo.models.reward_model.hps_score import HPSClipRewardModel
from fastvideo.models.reward_model.clip_score import CLIPScoreRewardModel
from fastvideo.models.reward_model.aes_score import Aesthetic_PredictorModel
from fastvideo.models.reward_model.utils import compute_reward
from PIL import Image
import json
from tqdm import tqdm

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

def img_path_to_prompt(img_path: Union[str, Path]) -> str:
    return os.path.basename(img_path).replace(".jpg", "").replace(".png", "").replace(".jpeg", "")

class ImgDataset(Dataset):
    def __init__(self, file):
        self.data = None
        with open(file, 'r') as f:
            self.data = json.load(f)
        
        for i, item in enumerate(self.data):
            if "image" not in item or "prompt" not in item:
                raise ValueError(f"Each item in the JSON file must contain 'image' and 'prompt' keys. Found: {item}")
            try:
                father_dir = os.path.dirname(file)
                relative_path = os.path.join(
                    father_dir,
                    os.path.basename(os.path.dirname(item["image"])),
                    os.path.basename(item["image"]),
                )
                #print( father_dir, os.path.basename(os.path.dirname(item["image"])), os.path.basename(item["image"]),)
                img = Path(relative_path)
                assert img.is_file(), f"Image file {img} does not exist."
            except:
                father_dir = os.path.dirname(file)
                relative_path = os.path.join(
                    father_dir,
                    os.path.basename(item["image"]),
                )
                #print( father_dir, os.path.basename(os.path.dirname(item["image"])), os.path.basename(item["image"]),)
                img = Path(relative_path)
                assert img.is_file(), f"Image file {img} does not exist."

            self.data[i]["index"] = i
            self.data[i]["image"] = relative_path

        print(f"### Loaded {len(self.data)} items from {file}")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def distributed_setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main(args):
    rank, local_rank, world_size = distributed_setup()
    ################################### Build Reward Model ###################################
    reward_models = []
    if args.reward_model == "hpsv2":
        reward_models.append(HPSClipRewardModel(
            device=f"cuda:{local_rank}",
            clip_ckpt_path=args.hps_clip_path,
            hps_ckpt_path=args.hps_path,
        ))
    elif args.reward_model == "image_reward":
        reward_models.append(ImageRewardModel(
            model_name=args.image_reward_path,
            device=f"cuda:{local_rank}",
            med_config=args.image_reward_med_config,
            http_proxy=args.image_reward_http_proxy,
            https_proxy=args.image_reward_https_proxy,
        ))
    elif args.reward_model == "clip_score":
        reward_models.append(CLIPScoreRewardModel(
            clip_model_path=args.clip_score_path,
            device=f"cuda:{local_rank}",
        ))
    elif args.reward_model == "pick_score":
        reward_models.append(PickScoreRewardModel(
            device=f"cuda:{local_rank}",
            http_proxy=args.pick_score_http_proxy,
            https_proxy=args.pick_score_https_proxy,
        ))
    elif args.reward_model == "unified_reward":
        unified_reward_urls = args.unified_reward_url.split(",")
        if isinstance(unified_reward_urls, list):
            num_urls = len(unified_reward_urls)
            ur_url_idx = rank % num_urls
            ur_url = unified_reward_urls[ur_url_idx]
            print(f"Rank {rank} using unified-reward URL: {ur_url}")
        reward_models.append(UnifiedRewardModel(
            model_name = "UnifiedRewardModel_semantic",
            api_url=ur_url,
            default_question_type=args.unified_reward_default_question_type,
            num_workers=args.unified_reward_num_workers,
        ))
    elif args.reward_model == "all":
        reward_models = [
            HPSClipRewardModel(
                device=f"cuda:{local_rank}",
                clip_ckpt_path=args.hps_clip_path,
                hps_ckpt_path=args.hps_path,
            ),
            ImageRewardModel(
                model_name=args.image_reward_path,
                device=f"cuda:{local_rank}",
                med_config=args.image_reward_med_config,
                http_proxy=args.image_reward_http_proxy,
                https_proxy=args.image_reward_https_proxy,
            ),
            CLIPScoreRewardModel(
                clip_model_path=args.clip_score_path,
                device=f"cuda:{local_rank}",
            ),
            PickScoreRewardModel(
                device=f"cuda:{local_rank}",
                http_proxy=args.pick_score_http_proxy,
                https_proxy=args.pick_score_https_proxy,
            ),
            Aesthetic_PredictorModel(
                model_name = "Aesthetic_PredictorModel",
                device=f"cuda:{local_rank}",
            )
        ]
        if args.unified_reward_url is not None:
            unified_reward_urls = args.unified_reward_url.split(",")
            if isinstance(unified_reward_urls, list):
                num_urls = len(unified_reward_urls)
                ur_url_idx = rank % num_urls
                ur_url = unified_reward_urls[ur_url_idx]
                print(f"Rank {rank} using unified-reward URL: {ur_url}")
            reward_models.append(
                UnifiedRewardModel(
                    model_name = "UnifiedRewardModel_semantic",
                    api_url=ur_url,
                    default_question_type=args.unified_reward_default_question_type,
                    num_workers=args.unified_reward_num_workers,
                )
            )
            reward_models.append(
                UnifiedRewardModel(
                    model_name = "UnifiedRewardModel_score",
                    api_url=ur_url,
                    default_question_type='score',
                    num_workers=args.unified_reward_num_workers,
                )
            )
    else:
        raise ValueError(f"Unsupported reward model: {args.reward_model}")

    #################################### Evaluate ###################################
    reward_results = []
    if args.single_img and args.single_img_prompt:
        prompt = args.single_img_prompt
        img_pil = Image.open(args.single_img).convert("RGB")

        _, _, rewards_dict, _ = compute_reward(
            images=[img_pil],
            input_prompts=[prompt],
            reward_models=reward_models,
            reward_weights={
                type(reward_model).__name__: 1.0 for reward_model in reward_models
            }
        )
        
        print(f"Single image evaluation completed: {args.single_img}")
        for model_name, rwd in rewards_dict.items():
            print(f"{model_name} Reward: {rwd[0]}")
        
    else:
        # dataset
        dataset = ImgDataset(args.prompt_file)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=6,
            pin_memory=True,
            drop_last=False
        )

        reward_results = []
        success_results = []
        for batch in tqdm(dataloader, desc=f"Rank {rank} Processing Batches", disable=rank != 0):
            index = batch["index"]
            img = batch["image"]
            prompt = batch["prompt"]

            img_pil = [Image.open(p).convert("RGB") for p in img]
            
            _, merged_successes, rewards_dict, _ = compute_reward(
                images=img_pil,
                input_prompts=prompt,
                reward_models=reward_models,
                reward_weights={
                    type(reward_model).__name__: 1.0 for reward_model in reward_models
                }
            )

            for i, (idx, img_path, prmpt) in enumerate(zip(index, img, prompt)):
                reward_meta = {
                    "index": idx.item() if isinstance(idx, torch.Tensor) else idx,
                    "image": img_path,
                    "reward": {}
                }
                for model_name, rwd in rewards_dict.items():
                    if model_name == "PickScoreRewardModel":
                        reward_meta["reward"][model_name] = (rwd[i] * 8.0 + 18.0) / 100.0
                    else:
                        reward_meta["reward"][model_name] = rwd[i]              
                reward_results.append(reward_meta)
                success_results.append(merged_successes[i])

        # Gather results from all ranks
        all_rewards = [None] * world_size
        dist.all_gather_object(all_rewards, reward_results)
        reward_results = []
        for rank_rewards in all_rewards:
            reward_results.extend(rank_rewards)
        all_successes = [None] * world_size
        dist.all_gather_object(all_successes, success_results)
        success_results = []
        for rank_successes in all_successes:
            success_results.extend(rank_successes)

        # Save results
        output_dir = os.path.dirname(args.output_json)
        os.makedirs(output_dir, exist_ok=True)
        if rank == 0:
            with open(args.output_json, "w") as f:
                json.dump(reward_results, f, indent=4)
            print(f"Rewards saved to {args.output_json}")
        else:
            print(f"Rank {rank} completed evaluation, but not saving results.")
        
        # check alternative and consistency
        assert len(reward_results) == len(dataset), f"Mismatch in number of reward results and dataset length: {len(reward_results)} vs {len(dataset)}"
        assert len(success_results) == len(reward_results), f"Mismatch in number of success results and reward results: {len(success_results)} vs {len(reward_results)}"
        indices = [item["index"] for item in reward_results]
        assert len(indices) == len(set(indices)), "Indices in reward results are duplicated."
        
        # save mean reward
        if rank == 0:
            save_mean = ""
            reward_mean = {}
            num_success = sum(success_results)
            save_mean += f"Total Successful Samples: {num_success}\n"
            for model_name, _ in rewards_dict.items():
                reward_mean[model_name] = np.mean([item["reward"][model_name] for j, item in enumerate(reward_results) if success_results[j] != 0])
                save_mean += f"{model_name} Mean Reward: {reward_mean[model_name]}\n"
            with open(args.output_json.replace(".json", "_mean.txt"), "w") as f:
                f.write(save_mean)
            print(save_mean)

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux Evlatuation Script")

    # Input Arguments
    parser.add_argument("--output_json", type=str, default="reward.json",
                        help="Path to save the output JSON file with rewards") 
    parser.add_argument("--single_img", type=str, default=None,
                        help="Path to a single image for evaluation")
    parser.add_argument("--single_img_prompt", type=str, default=None,
                        help="Prompt for the single image if not derived from filename")
    parser.add_argument("--prompt_file", type=str, default="prompts.json"
                        , help="Path to the JSON file containing prompts for images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")

    # Reward Model Arguments
    parser.add_argument("--reward_model", type=str, default="ImageReward",
        help="Type of reward model to use")
    parser.add_argument("--hps_clip_path", type=str, default=None,
                        help="Path to the HPS clip model checkpoint")
    parser.add_argument("--hps_path", type=str, default=None,
                        help="Path to the HPS model checkpoint")
    parser.add_argument("--image_reward_path", type=str, default=None,
                        help="Path to the ImageReward model")
    parser.add_argument("--image_reward_med_config", type=str, default=None,
                        help="Path to the ImageReward model config")
    parser.add_argument("--image_reward_http_proxy", type=str, default=None,
                        help="HTTP proxy for ImageReward model")
    parser.add_argument("--image_reward_https_proxy", type=str, default=None,
                        help="HTTPS proxy for ImageReward model")
    parser.add_argument("--clip_score_path", type=str, default=None,
                        help="Path to the CLIPScore model")
    parser.add_argument("--pick_score_http_proxy", type=str, default=None,
                        help="HTTP proxy for PickScore model")
    parser.add_argument("--pick_score_https_proxy", type=str, default=None,
                        help="HTTPS proxy for PickScore model")
    parser.add_argument("--unified_reward_url", type=str, default=None,
                        help="URL for the UnifiedReward model")
    parser.add_argument("--unified_reward_default_question_type", type=str, default="default",
                        help="Default question type for UnifiedReward model")
    parser.add_argument("--unified_reward_num_workers", type=int, default=1,
                        help="Number of workers for UnifiedReward model")

    args = parser.parse_args()

    if args.image_reward_http_proxy == "None":
        args.image_reward_http_proxy = None
    if args.image_reward_https_proxy == "None":
        args.image_reward_https_proxy = None
    if args.pick_score_http_proxy == "None":
        args.pick_score_http_proxy = None
    if args.pick_score_https_proxy == "None":
        args.pick_score_https_proxy = None
    if args.unified_reward_url == "None":
        args.unified_reward_url = None

    main(args)
