# __init__.py
from . import tracker
from . import expand_mask_dir

WEB_DIRECTORY = "./web"

# Correctly merge the mappings from all node files
NODE_CLASS_MAPPINGS = {
    **tracker.NODE_CLASS_MAPPINGS,
    **expand_mask_dir.NODE_CLASS_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **tracker.NODE_DISPLAY_NAME_MAPPINGS,
    **expand_mask_dir.NODE_DISPLAY_NAME_MAPPINGS
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### Loading: ComfyUI-TJNodes")