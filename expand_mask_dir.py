import torch
import numpy as np

class ExpandMaskDir:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_up": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "expand_down": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "expand_mask_dir"
    CATEGORY = "mask/transform" # You can change this category if you like

    def expand_mask_dir(self, mask, expand_left, expand_right, expand_up, expand_down):
        if mask.dim() == 2:
            # Add batch dimension if missing (e.g., coming from MaskPreview node)
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:
             # Remove channel dimension if present (e.g., coming from VAE Decode)
             mask = mask.squeeze(1)
        
        if mask.dim() != 3:
             raise ValueError(f"Input mask has unexpected dimensions: {mask.shape}")

        batch_size, height, width = mask.shape
        # Create output tensor initialized to zeros (black)
        mask_out = torch.zeros_like(mask)

        for i in range(batch_size):
            single_mask = mask[i]
            
            # Find coordinates of non-zero pixels (white areas)
            # Using cpu() for nonzero as it might be faster for potentially sparse masks
            nz = torch.nonzero(single_mask.cpu()) # Returns shape (num_nonzero, 2) -> (y, x)

            if nz.shape[0] == 0: 
                # If the input mask is empty, the output is also empty
                continue 

            # Determine bounding box of the white area
            min_y = torch.min(nz[:, 0])
            max_y = torch.max(nz[:, 0])
            min_x = torch.min(nz[:, 1])
            max_x = torch.max(nz[:, 1])

            # Calculate new bounding box coordinates, clamping to image dimensions
            # Ensure coordinates are integers
            new_min_y = max(0, int(min_y) - expand_up)
            new_max_y = min(height - 1, int(max_y) + expand_down)
            new_min_x = max(0, int(min_x) - expand_left)
            new_max_x = min(width - 1, int(max_x) + expand_right)

            # Fill the expanded area in the output mask for the current batch item
            # Ensure slicing uses integers and includes the max boundary
            mask_out[i, new_min_y:new_max_y+1, new_min_x:new_max_x+1] = 1.0

        return (mask_out,)

# Mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ExpandMaskDir": ExpandMaskDir
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExpandMaskDir": "Expand Mask Direction"
}