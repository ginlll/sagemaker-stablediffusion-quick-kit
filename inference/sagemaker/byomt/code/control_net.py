import os
import json
import uuid
import io
import sys
import tarfile
import traceback

from PIL import Image

import torch
from transformers import pipeline as depth_pipeline
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector,MLSDdetector,HEDdetector,HEDdetector

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
from diffusers import UniPCMultistepScheduler



import cv2
import numpy as np


control_net_postfix=[
                                    "canny",
                                    "openpose"
                                ]

def convert_lora(pipeline, checkpoint_path, LORA_PREFIX_UNET, LORA_PREFIX_TEXT_ENCODER, alpha):

    # load LoRA weight from .safetensors

    state_dict = load_file(checkpoint_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
        return pipeline


class ControlNetDectecProcessor:
    def __init__(self):
        self.openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        self.depth= depth_pipeline('depth-estimation')
        
    
    def detect_process(self,model_name,image_url):
        if model_name not in control_net_postfix:
            return None
        func = getattr(ControlNetDectecProcessor, f'get_{model_name}_image')
        return func(self,image_url)
    
        
    def get_openpose_image(self,image_url):
        image = load_image(image_url)
        pose_image = self.openpose(image)
        return pose_image

    def get_mlsd_image(self,image_url):
        image = load_image(image_url)
        mlsd_image = self.mlsd(image)
        return mlsd_image

    def get_hed_image(self,image_url):
        image = load_image(image_url)
        hed_image = self.hed(image)
        return hed_image

    def get_scribble_image(self,image_url):
        image = load_image(image_url)
        scribble_image = self.hed(image,scribble=True)
        return scribble_image

    def get_depth_image(self,image_url):
        image = load_image(image_url)
        depth_image = self.depth(image)['depth']
        depth_image = np.array(depth_image)
        depth_image = depth_image[:, :, None]
        depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
        depth_image = Image.fromarray(depth_image)
        return depth_image

    def get_canny_image(self,image_url):
        image = load_image(image_url)
        image = np.array(image)
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image


def init_control_net_model():
    print(f"init_control_net_model:{control_net_postfix} begain")
    for model in control_net_postfix:
        controlnet = ControlNetModel.from_pretrained(
                                    f"lllyasviel/sd-controlnet-{model}", torch_dtype=torch.float16
                            )
    print(f"init_control_net_model:{control_net_postfix} completed")
    
def init_control_net_pipeline(base_model,control_net_model):
    if control_net_model not in control_net_postfix:
            return None
    controlnet = ControlNetModel.from_pretrained(
                                    f"lllyasviel/sd-controlnet-{control_net_model}", torch_dtype=torch.float16
                            )
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    convert_lora(model_name, f"/tmp/TDX.1", 'lora_unet', 'lora_te', 0.7)
    convert_lora(model_name, f"/tmp/charturnerbetaLora_charturnbetalora", 'lora_unet', 'lora_te', 0.3)
    pipeline.save_pretrained(f"/tmp/mix_model")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
                                f"/tmp/mix_model", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
                                )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


        
    
    return pipe