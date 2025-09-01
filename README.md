# 🖼️ Text-based Image Editing with SAM + Stable Diffusion

This project demonstrates a **text-based image editing pipeline** using:
- **GroundingDINO + SAM (Segment Anything Model)** for **text-prompted segmentation**
- **Stable Diffusion Inpainting** for **content-aware image editing**

---

## 🚀 Overview

Given an input **image** and a **text prompt** (e.g., `"hands"`), the pipeline works as follows:

1. **Segmentation with Text Prompt**
   - A region of interest is segmented using **GroundingDINO + SAM**, which identifies the target object (e.g., hands, face, shoes).
   
2. **Mask-based Editing**
   - The segmented mask is passed into a **Stable Diffusion Inpainting pipeline**, along with a **new text prompt** describing how to edit the masked area (e.g., `"replace hands with gloves"`).

3. **Final Output**
   - A new edited image is generated that preserves the unmasked regions while modifying only the target region.

---

## 📊 Example Results

Below are some examples of the pipeline outputs (placeholders — replace with your images):

| Input Image | Segmentation Mask | Edited Output |
|-------------|-------------------|---------------|
| ![Input](examples/input1.jpg) | ![Mask](examples/mask1.png) | ![Output](examples/output1.jpg) |
| ![Input](examples/input2.jpg) | ![Mask](examples/mask2.png) | ![Output](examples/output2.jpg) |

---

## 💡 Potential Use Cases

- ✨ **Creative Editing**: Modify specific parts of an image using natural language prompts.  
- 🖼️ **Artistic Workflows**: Artists/designers can replace or enhance elements quickly.  
- 📷 **Photo Retouching**: Edit unwanted objects or enhance details without manual masking.  
- 🧪 **Research**: Study controllable image editing and prompt-based manipulation.  

---

## ⚙️ Project Structure

