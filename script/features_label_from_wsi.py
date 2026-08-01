from pathlib import Path
import openslide
import cv2
import numpy as np
import torch

def extract_features_from_wsi(file_path, patch_size=512, level=1, feature_extractor=None, transform=None, device='cuda'):
    slide_name = Path(file_path).stem
    slide = openslide.OpenSlide(str(file_path))
    downsample = slide.level_downsamples
    slide_dimension = slide.level_dimensions

    discarded_patches = 0
    saved_patches = 0

    features = []
    patch_metadata = []

    # Calculate coordinates
    coord_x = list(range(0, slide_dimension[level][0] - patch_size, patch_size)) 
    coord_x.append(slide_dimension[level][0] - patch_size) 
    
    coord_y = list(range(0, slide_dimension[level][1] - patch_size, patch_size))
    coord_y.append(slide_dimension[level][1] - patch_size)

    for x in coord_x:
        for y in coord_y:
            # 1. Read region from WSI
            patch = slide.read_region(
                (int(x * downsample[level]), int(y * downsample[level])), 
                level, 
                (patch_size, patch_size)
            )

            patch_rgb = patch.convert("RGB")
            patch_np = np.array(patch_rgb)
            patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
            
            # Access HSV channels
            h = patch_hsv[:, :, 0]  # Hue
            s = patch_hsv[:, :, 1]  # Saturation
            v = patch_hsv[:, :, 2]  # Value (brightness)

            # 2. Tissue detection (background usually > 60%)
            pink = (h > 160) & (h < 180)
            purple = (h > 120) & (h < 160)
            tissue_pink = pink & (s > 10)
            tissue_purple = purple & (s > 8)
            tissue = tissue_pink | tissue_purple
            check_tissue = np.mean(tissue) > 0.4

            # 3. Artifact detection
            other_color = (~tissue) & (s > 30)
            check_no_artifact = np.mean(other_color) < 0.05

            check_img = check_tissue and check_no_artifact

            # 4. Feature Extraction if patch is valid
            if check_img:
                patch_tensor = transform(patch_rgb)
                patch_tensor = patch_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = feature_extractor(patch_tensor)
                
                feature = out['features'].squeeze()
                features.append(feature)
                
                # Save metadata
                patch_metadata.append({
                    "slide_id": slide_name,
                    "level": level,
                    "height": patch_size,
                    "width": patch_size,
                    "origin_x": x,
                    "origin_y": y
                })
                saved_patches += 1
            else:
                discarded_patches += 1
                
    if saved_patches == 0:
        slide.close()
        return None, [], saved_patches, discarded_patches
        
    feature_tensor = torch.stack(features) # Tensor with all extracted features
    slide.close()
    
    return feature_tensor, patch_metadata, saved_patches, discarded_patches