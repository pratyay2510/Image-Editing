import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import (
    SamModel, SamProcessor,
    AutoProcessor, AutoModelForZeroShotObjectDetection
)

class BoundingBoxSegmentationPipeline:
    def __init__(self,
                 sam_model_id: str = "./sam_model",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Bounding box-guided segmentation using SAM.
        
        Args:
            sam_model_id: Path to SAM model
            device: Device to run inference on
        """
        self.device = device
        
        # Load SAM
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(device)
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        
    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Load and convert image to PIL format."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 3:
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Image numpy array must be HxWx3")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError("Image must be str / ndarray / PIL.Image")
    
    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Process mask to ensure it's 2D binary format.
        
        Args:
            mask: Input mask that may have extra dimensions
            
        Returns:
            2D binary mask (H, W)
        """
        # Handle different mask shapes
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                # Shape: (1, H, W) -> (H, W)
                mask = mask.squeeze(0)
            elif mask.shape[-1] == 1:
                # Shape: (H, W, 1) -> (H, W)
                mask = mask.squeeze(-1)
            else:
                # Shape: (C, H, W) -> take first channel -> (H, W)
                mask = mask[0]
        elif mask.ndim == 4:
            # Shape: (1, C, H, W) or (C, 1, H, W) -> (H, W)
            mask = mask.squeeze()
            if mask.ndim == 3:
                mask = mask[0]
        
        # Ensure binary values
        mask = (mask > 0.5).astype(np.uint8)
        
        return mask
    
    def segment_with_box(self,
                        image: Union[str, np.ndarray, Image.Image],
                        box: Union[List, Tuple, np.ndarray],
                        multimask_output: bool = False,
                        show_result: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Segment image region based on bounding box coordinates.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            box: Bounding box coordinates as [x_min, y_min, x_max, y_max] 
                 where (x_min, y_min) is top-left corner and (x_max, y_max) is bottom-right corner
            multimask_output: Whether to return multiple mask candidates
            show_result: Whether to display the segmentation result
            
        Returns:
            Segmentation mask (HxW numpy array) or list of masks if multimask_output=True
        """
        # Load and validate image
        pil_img = self._load_image(image)
        
        # Validate box coordinates
        if len(box) != 4:
            raise ValueError("Box must contain exactly 4 coordinates: [x_min, y_min, x_max, y_max]")
        
        x_min, y_min, x_max, y_max = box
        
        # Ensure coordinates are within image bounds
        img_width, img_height = pil_img.size
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))
        
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Invalid bounding box: x_min must be < x_max and y_min must be < y_max")
        
        box_coords = [x_min, y_min, x_max, y_max]
        print(f"📦 Processing bounding box: [{x_min}, {y_min}, {x_max}, {y_max}]")
        
        # Prepare inputs for SAM
        sam_inputs = self.sam_processor(
            pil_img, 
            input_boxes=[[box_coords]], 
            return_tensors="pt"
        ).to(self.device)
        
        # Run SAM inference
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)
        
        # Post-process masks
        masks = self.sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )
        
        if multimask_output:
            # Return all mask candidates
            masks_list = []
            for mask_tensor in masks[0]:
                mask_np = mask_tensor.numpy()
                mask_processed = self._process_mask(mask_np)
                masks_list.append(mask_processed)
            
            print(f"✅ Generated {len(masks_list)} mask candidates")
            
            if show_result:
                self._display_result(pil_img, masks_list, box_coords, multimask_output=True)
            
            return masks_list
        else:
            # Return single best mask
            mask_tensor = masks[0][0]  # First mask from first batch
            mask_np = mask_tensor.numpy()
            mask_processed = self._process_mask(mask_np)
            
            print("✅ Generated segmentation mask")
            print(f"   Mask shape: {mask_processed.shape}")
            
            if show_result:
                self._display_result(pil_img, [mask_processed], box_coords, multimask_output=False)
            
            return mask_processed
    
    def segment_multiple_boxes(self,
                              image: Union[str, np.ndarray, Image.Image],
                              boxes: List[Union[List, Tuple]],
                              show_result: bool = True) -> List[np.ndarray]:
        """
        Segment multiple regions using multiple bounding boxes.
        
        Args:
            image: Input image
            boxes: List of bounding boxes, each as [x_min, y_min, x_max, y_max]
            show_result: Whether to display results
            
        Returns:
            List of segmentation masks (one per box)
        """
        pil_img = self._load_image(image)
        all_masks = []
        
        print(f"📦 Processing {len(boxes)} bounding boxes...")
        
        for i, box in enumerate(boxes):
            print(f"🔄 Processing box {i+1}/{len(boxes)}: {box}")
            mask = self.segment_with_box(pil_img, box, show_result=False)
            all_masks.append(mask)
        
        print(f"✅ Generated {len(all_masks)} masks")
        
        if show_result:
            self._display_multiple_results(pil_img, all_masks, boxes)
        
        return all_masks
    
    def _display_result(self, 
                       pil_img: Image.Image, 
                       masks: List[np.ndarray], 
                       box_coords: List[int], 
                       multimask_output: bool = False):
        """Display segmentation results."""
        num_plots = len(masks) + 2  # Original + box + masks
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        # Handle single subplot case
        if num_plots == 3:
            if not hasattr(axes, '__len__'):
                axes = [axes]
        
        # Original image
        axes[0].imshow(pil_img)
        axes[0].set_title("Original Image")
        # axes[0].axis('off')
        
        # Original with bounding box
        axes[1].imshow(pil_img)
        x_min, y_min, x_max, y_max = box_coords
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=3, edgecolor='lime', facecolor='none'
        )
        axes[1].add_patch(rect)
        axes[1].set_title("Bounding Box")
        axes[1].axis('off')
        
        # Display masks
        for i, mask in enumerate(masks):
            title = f"Mask {i+1}" if multimask_output else "Segmentation Mask"
            
            # Debug: Print mask info
            print(f"   Displaying mask {i+1}: shape={mask.shape}, dtype={mask.dtype}, range=[{mask.min()}, {mask.max()}]")
            
            # Ensure mask is 2D
            if mask.ndim != 2:
                print(f"   Warning: Mask {i+1} has unexpected shape {mask.shape}, attempting to fix...")
                mask = self._process_mask(mask)
            
            axes[i + 2].imshow(mask, cmap='gray')
            axes[i + 2].set_title(title)
            axes[i + 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _display_multiple_results(self, 
                                 pil_img: Image.Image, 
                                 masks: List[np.ndarray], 
                                 boxes: List):
        """Display results for multiple bounding boxes."""
        num_boxes = len(boxes)
        if num_boxes == 0:
            return
            
        fig, axes = plt.subplots(2, num_boxes + 1, figsize=(5 * (num_boxes + 1), 10))
        
        # Handle single column case
        if num_boxes == 1:
            axes = axes.reshape(2, -1)
        
        # Top row: Original image with all boxes
        axes[0, 0].imshow(pil_img)
        axes[0, 0].set_title("All Bounding Boxes")
        
        # Draw all bounding boxes on the first image
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=plt.cm.tab10(i), facecolor='none'
            )
            axes[0, 0].add_patch(rect)
        axes[0, 0].axis('off')
        
        # Individual boxes and masks
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            # Top row: Individual box
            axes[0, i + 1].imshow(pil_img)
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=3, edgecolor='lime', facecolor='none'
            )
            axes[0, i + 1].add_patch(rect)
            axes[0, i + 1].set_title(f"Box {i+1}")
            axes[0, i + 1].axis('off')
            
            # Bottom row: Corresponding mask
            # Ensure mask is 2D
            if mask.ndim != 2:
                mask = self._process_mask(mask)
            
            axes[1, i + 1].imshow(mask, cmap='gray')
            axes[1, i + 1].set_title(f"Mask {i+1}")
            axes[1, i + 1].axis('off')
        
        # Bottom left: Combined masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            if mask.ndim != 2:
                mask = self._process_mask(mask)
            combined_mask = np.maximum(combined_mask, mask)
        
        axes[1, 0].imshow(combined_mask, cmap='gray')
        axes[1, 0].set_title("Combined Masks")
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def __call__(self,
                 image: Union[str, np.ndarray, Image.Image],
                 box: Union[List, Tuple, np.ndarray] = None,
                 multimask_output: bool = False,
                 show_result: bool = True):
        """
        Main call method for single bounding box segmentation.
        
        Args:
            image: Input image
            box: Bounding box coordinates [x_min, y_min, x_max, y_max]
            multimask_output: Return multiple mask candidates
            show_result: Display results
            
        Returns:
            Segmentation mask(s)
        """
        if box is None:
            raise ValueError("Bounding box coordinates must be provided")
        
        return self.segment_with_box(image, box, multimask_output, show_result)







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
