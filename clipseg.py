import torch
import numpy as np
import cv2
from typing import Union
from PIL import Image
import os
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

from transformers import (
    SamModel, SamProcessor,
    AutoProcessor, AutoModelForZeroShotObjectDetection
)
from diffusers import StableDiffusionInpaintPipeline
import torch




def create_pipe(model):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype = torch.float16)
    pipe = pipe.to('cuda:0')
    return pipe

class TextSegmentationPipeline:
    def __init__(self,
                 dino_model_id: str = "./dino_model",
                 sam_model_id: str = "./sam_model",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Text-prompt-based segmentation using GroundingDINO + SAM.
        """
        self.device = device

        # Load GroundingDINO
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

        # Load SAM
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(device)
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)

    def __call__(self,
                 image: Union[str, np.ndarray, Image.Image],
                 text_prompt: str,
                 box_threshold: float = 0.3,
                 text_threshold: float = 0.25):
        """
        Args:
            image: path, np.ndarray (H,W,3 BGR/RGB), or PIL.Image
            text_prompt: text query (e.g., "hands")
            box_threshold: confidence for box filtering
            text_threshold: confidence for text filtering
        Returns:
            mask (numpy array HxW, binary)
        """
        # ---- Load image ----
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Image numpy array must be HxWx3")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError("Image must be str / ndarray / PIL.Image")

        # ---- 1. Get boxes from GroundingDINO ----
        dino_inputs = self.dino_processor(images=pil_img, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            dino_outputs = self.dino_model(**dino_inputs)

        # Get raw boxes + scores
        target_sizes = torch.tensor([pil_img.size[::-1]])  # (h, w)
        results = self.dino_processor.post_process_grounded_object_detection(
            dino_outputs,
            dino_inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes
        )

        boxes = results[0]["boxes"].cpu().numpy()  # xyxy format
        if len(boxes) == 0:
            raise ValueError(f"No object found for prompt '{text_prompt}'")

        # ---- 2. Get mask from SAM ----
        # We’ll just take the first box for simplicity
        box = boxes[0]

        sam_inputs = self.sam_processor(pil_img, input_boxes=[[box.tolist()]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        masks = self.sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )

        mask = masks[0][0].numpy().astype(np.uint8)  # binary mask

        return mask


