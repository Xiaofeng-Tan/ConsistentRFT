# Image-Reward: Copyied from https://github.com/THUDM/ImageReward
import os
from typing import Union, List
from PIL import Image

import torch
try:
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
except:
    raise Warning("ImageReward is required to be installed (`pip install image-reward`) when using ImageReward for post-training.")


class Aesthetic_PredictorModel(object):
    def __init__(self, model_name, device, http_proxy=None, https_proxy=None, med_config=None):
        if http_proxy:
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["https_proxy"] = https_proxy
        self.model_name = model_name if model_name else "Aesthetic_PredictorModel-v1.0"
        self.device = device
        self.med_config = med_config
        self.build_reward_model()

    def build_reward_model(self):
        model, preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = model.to(dtype = torch.bfloat16, device=self.device)
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(
            self,
            images,
            texts,
    ):
        if isinstance(texts, str):
            texts = [texts] * len(images)
        
        rewards = []
        for image, text in zip(images, texts):
            ranking, reward = self.model.inference_rank(text, [image])
            rewards.append(reward)
        return rewards

    @torch.no_grad()
    def __call__(self, images, texts=None):
        """
        Args:
            images: List of PIL.Image objects
            texts: 占位参数
        Returns:
            List[float]: aesthetics scores
        """
        pixel_values = self.preprocessor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)

        logits = self.model(pixel_values).logits.squeeze()
        scores = logits.float().cpu().numpy()

        if scores.ndim == 0:  # 单张图
            scores = [float(scores)]
        else:  # 多张图
            scores = scores.tolist()
        return scores
