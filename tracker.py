# tracker.py

import torch
import numpy as np
import subprocess
import sys
import os
import time
import imageio
from aiohttp import web
import server
import random
import string
import folder_paths

# --- Global dictionary to hold the filename for each node instance ---
NODE_FILENAME_STATE = {}

# Get the directory of the current script to find tracker_gui.py
current_dir = os.path.dirname(os.path.realpath(__file__))
gui_script_path = os.path.join(current_dir, "tracker_gui.py")

@server.PromptServer.instance.routes.get("/tj-nodes/launch-tracker")
async def launch_tracker_gui_route(request):
    """
    This function launches the tracker GUI and stores the expected filename.
    """
    if 'node_id' not in request.query:
        return web.json_response({"status": "error", "message": "node_id is required"}, status=400)
    
    node_id = request.query['node_id']
    
    try:
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        output_filename = f"Mask_{random_str}.mp4"
        output_path = folder_paths.get_temp_directory()
        
        NODE_FILENAME_STATE[node_id] = output_filename
        
        print(f"[Tracker Node] Stored state for node {node_id}: {output_filename}")
        
        subprocess.Popen([sys.executable, gui_script_path, output_path, output_filename])
        
        return web.json_response({
            "status": "success",
            "filename": output_filename,
        })
    except Exception as e:
        print(f"ERROR: Failed to launch Tracker GUI: {e}")
        return web.json_response({"status": "error", "message": str(e)}, status=500)

class Tracker:
    @classmethod
    def IS_CHANGED(s, unique_id, **kwargs):
        return random.random()

    @classmethod
    def INPUT_TYPES(s):
        return {
            # The button is now created entirely in the JavaScript file.
            "required": {},
            "hidden": { "prompt": "PROMPT", "unique_id": "UNIQUE_ID" },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "Tracking"

    def execute(self, prompt=None, unique_id=None):
        if unique_id not in NODE_FILENAME_STATE:
            raise Exception("Tracker node has not been run. Please click 'Track Video' and finish the process in the GUI before queueing the prompt.")
        
        filename = NODE_FILENAME_STATE[unique_id]
        video_path = os.path.join(folder_paths.get_temp_directory(), filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Mask file not found. Please ensure you 'Export to ComfyUI' from the tracker GUI before queueing the prompt. Expected file: {video_path}")

        print(f"[Tracker Node Execute] File found. Loading video: {video_path}")
        
        try:
            reader = imageio.get_reader(video_path)
            frames = [np.array(frame) for frame in reader]
            reader.close()
            
            if not frames:
                raise ValueError(f"Video file '{filename}' is empty or could not be read.")

            video_data = np.stack(frames).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(video_data)
            
            mask_tensor = image_tensor[:, :, :, 0]

            return (image_tensor, mask_tensor,)
        except Exception as e:
            print(f"ERROR: Failed to load video from {video_path}: {e}")
            raise

NODE_CLASS_MAPPINGS = {"Tracker": Tracker}
NODE_DISPLAY_NAME_MAPPINGS = {"Tracker": "Point Tracker ✨"}
