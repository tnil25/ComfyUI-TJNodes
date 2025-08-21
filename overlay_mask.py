import torch
import numpy as np
from PIL import Image

# This class definition replaces your old one.
class OverlayMask:
    """
    A node to overlay a mask onto an image with a specified color and opacity.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("COLOR", {"default": "#FF0000"}), # Or "STRING" if not using the JS widget
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay_mask"
    CATEGORY = "mask/compositing" # Or your preferred category

    # Helper functions remain the same
    def hex_to_rgb(self, hex_color: str):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def tensor_to_pil(self, tensor):
        image_np = tensor.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np, 'RGB')

    def pil_to_tensor(self, pil_image):
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)

    def overlay_mask(self, image: torch.Tensor, mask: torch.Tensor, color: str, opacity: float):
        output_images = []
        rgb_color = self.hex_to_rgb(color)

        for img_tensor, mask_tensor in zip(image, mask):
            base_image_pil = self.tensor_to_pil(img_tensor)
            mask_np = mask_tensor.cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), 'L')
            
            # <<< --- NEW RESIZING LOGIC --- >>>
            image_size = base_image_pil.size
            if mask_pil.size != image_size:
                # Resize the mask. LANCZOS is a high-quality filter, good for masks.
                mask_pil = mask_pil.resize(image_size, Image.Resampling.LANCZOS)
            # <<< --- END OF NEW LOGIC --- >>>

            color_image_pil = Image.new("RGB", base_image_pil.size, rgb_color)
            base_image_pil = base_image_pil.convert("RGBA")
            color_image_pil = color_image_pil.convert("RGBA")
            
            alpha_np = np.array(mask_pil).astype(np.float32) * opacity
            alpha_channel = Image.fromarray(alpha_np.astype(np.uint8))
            
            color_image_pil.putalpha(alpha_channel)
            
            composite_image_pil = Image.alpha_composite(base_image_pil, color_image_pil)
            composite_image_pil = composite_image_pil.convert("RGB")

            output_tensor = self.pil_to_tensor(composite_image_pil)
            output_images.append(output_tensor)

        final_batch = torch.cat(output_images, dim=0)
        return (final_batch,)


# A dictionary that ComfyUI uses to register the nodes in this file.
NODE_CLASS_MAPPINGS = {
    "OverlayMaskNode": OverlayMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OverlayMaskNode": "Overlay Mask with Color"
}