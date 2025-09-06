import torch
import numpy as np
import cv2
from typing import Union
from PIL import Image
import os
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from diffusers import StableDiffusionInpaintPipeline
import torch


def create_pipe(model,device='cuda:0'):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype = torch.float16)
    pipe = pipe.to(device)
    return pipe







