import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import math
import sys
import threading
import queue
import torch
import imageio.v3 as iio # Not strictly needed if using cv2.VideoCapture to read frames
import psutil
import os
import subprocess


# --- Console Redirector Class ---
class ConsoleRedirector(object):
    """Redirects stdout to a Tkinter Label widget."""
    def __init__(self, widget):
        self.widget = widget # This will now be a ttk.Label
        self.stdout = sys.stdout # Keep a reference to original stdout

    def write(self, text):
        # Update the text of the Label directly
        # Strip trailing newlines for cleaner display in a single-line label
        clean_text = text.strip()
        if clean_text: # Only update if there's actual text to display
            self.widget.config(text=clean_text)
        
        # Also print to the original console for debugging purposes
        self.stdout.write(text)
        sys.stdout.flush() # Ensure it's printed immediately

    def flush(self):
        # This method is required for file-like objects
        self.stdout.flush()
        

# --- Main Application Class ---
class PointTrackerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("CoTracker")
        
        # --- Receive and store path and filename from ComfyUI ---
        self.output_path = None
        self.output_filename = None
        if len(sys.argv) >= 3:
            self.output_path = sys.argv[1]
            self.output_filename = sys.argv[2]
            print(f"GUI received output path: {self.output_path}")
            print(f"GUI received output filename: {self.output_filename}")
            filename_without_extension = os.path.splitext(self.output_filename)[0]
            self.master.title(f"CoTracker - {filename_without_extension}")
        else:
            print("Warning: Output information not provided by launcher.")

        # --- Device Configuration for CoTracker ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device} for CoTracker.")

        # --- CoTracker Model Initialization ---
        # We'll load the model when video is loaded, to allow for a loading indicator.
        self.cotracker_model = None

        # --- Video and Frame Variables ---
        self.video_path = None
        self.cap = None # OpenCV VideoCapture object
        self.total_frames = 0
        self.fps = 0
        self.display_width = 640 # Default display size
        self.display_height = 480 # Default display size
        self.video_frames_tensor = None # Stores the entire video as a PyTorch tensor (B T C H W)

        # --- Tracking Data ---
        self.selected_points_for_tracking = [] # [(x, y), ...] from user clicks
        self.tracked_points_cotracker = None # Stores CoTracker's pred_tracks (T N 2)
        self.tracked_visibility_cotracker = None # Stores CoTracker's pred_visibility (T N 1)
        self.current_display_point_coords = [] # For drawing on canvas
        self.current_display_point_visibility = [] # For drawing on canvas

        # Calculated position, scale, rotation based on tracked points
        self.tracked_midpoints = None # (T, 2)
        self.tracked_scales = None # (T,)
        self.tracked_rotations = None # (T,) (in degrees)

        # Initial state of tracked points for transformation pivots
        self.initial_tracked_midpoint_val = None
        self.initial_tracked_distance_val = None
        self.initial_tracked_angle_val = None

        # Last known good transformation values for mask persistence
        self.last_valid_midpoint = None
        self.last_valid_scale = 1.0
        self.last_valid_rotation_rad = 0.0

        # Mask Tracking Data
        self.initial_mask_centroid = None # Original coordinates of the mask centroid
        self.tracked_mask_centroid_coords = None # (T, 2) Tracked centroid of the mask
        self.initial_mask_polygon_for_export = None # Correctly initialized


# --- UI Elements ---
        # Initialize canvas with default size, it will be resized when video loads
        self.canvas = tk.Canvas(master, width=self.display_width, height=self.display_height, bg="black")
        self.canvas.pack(pady=10)

        # Slider
        self.slider = ttk.Scale(master, from_=0, to=0, orient=tk.HORIZONTAL,
                                command=self.slider_moved, state=tk.DISABLED)
        self.slider.pack(fill=tk.X, padx=10, pady=(5, 0)) # Padding top=5, bottom=0

        # Play/Pause button
        self.play_pause_button = ttk.Button(master, text="▶", command=self.toggle_play_pause, state=tk.DISABLED, width=12)
        self.play_pause_button.pack(pady=5)
        self.is_playing = False
        self.playback_job_id = None # For master.after loop

        # --- Main Button Frame (no longer contains the play/pause button) ---
        self.buttons_frame = ttk.Frame(master)
        self.buttons_frame.pack(pady=5)

        self.load_video_button = ttk.Button(self.buttons_frame, text="Load Video", command=self.load_video)
        self.load_video_button.pack(side=tk.LEFT, padx=5)

        self.select_points_button = ttk.Button(self.buttons_frame, text="Select Points", command=self.toggle_point_selection_mode, state=tk.DISABLED)
        self.select_points_button.pack(side=tk.LEFT, padx=5)
        self.is_selecting_points = False

        self.track_button = ttk.Button(self.buttons_frame, text="Track Points", command=self.start_tracking, state=tk.DISABLED)
        self.track_button.pack(side=tk.LEFT, padx=5)
        
        self.mask_button = ttk.Button(self.buttons_frame, text="Draw Mask", command=self.toggle_masking_mode, state=tk.DISABLED)
        self.mask_button.pack(side=tk.LEFT, padx=5)
        self.is_masking_mode = False
        self.mask_polygons = [] 
        self.current_mask_polygon_points = [] 
        self.is_moving_mask_point = False 
        self.moving_point_index = -1 
        self.mask_point_snap_radius = 8
        
        self.clear_button = ttk.Button(self.buttons_frame, text="Clear All", command=self.clear_tracking_and_points) # MODIFIED
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = ttk.Button(self.buttons_frame, text="Export As...", command=self.handle_export_normal, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        self.export_to_comfy_button = ttk.Button(self.buttons_frame, text="Export to ComfyUI", command=self.handle_export_to_comfy)
        self.export_to_comfy_button.pack(side=tk.LEFT, padx=5)

        # --- Checkboxes for Overlay Display ---
        self.overlay_options_frame = ttk.LabelFrame(master, text="Options")
        self.overlay_options_frame.pack(pady=5, padx=10, fill=tk.X)

        self.show_position = tk.BooleanVar(value=True)
        self.show_scale = tk.BooleanVar(value=False)
        self.show_rotation = tk.BooleanVar(value=False)

        # Command to redraw canvas when checkbox state changes
        # Use a lambda to pass arguments or ensure display_frame is called with current index
        self.position_checkbox = ttk.Checkbutton(self.overlay_options_frame, text="Position", variable=self.show_position,
                                                 command=lambda: self.display_frame(self.current_replay_index), state=tk.DISABLED)
        self.position_checkbox.pack(side=tk.LEFT, padx=5, pady=2)

        self.scale_checkbox = ttk.Checkbutton(self.overlay_options_frame, text="Scale", variable=self.show_scale,
                                              command=lambda: self.display_frame(self.current_replay_index), state=tk.DISABLED)
        self.scale_checkbox.pack(side=tk.LEFT, padx=5, pady=2)

        self.rotation_checkbox = ttk.Checkbutton(self.overlay_options_frame, text="Rotation", variable=self.show_rotation,
                                                 command=lambda: self.display_frame(self.current_replay_index), state=tk.DISABLED)
        self.rotation_checkbox.pack(side=tk.LEFT, padx=5, pady=2)


        # --- Console Output & Progress Bar ---
        self.console_label = ttk.Label(master, text="Load a video to begin.", wraplength=self.display_width, anchor="w", justify="left", font=("Arial", 10))
        self.console_label.pack(fill=tk.X, padx=10, pady=5)
        self.console_redirector = ConsoleRedirector(self.console_label) # Store reference
        sys.stdout = self.console_redirector # Redirect stdout

        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=self.display_width, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        self.progress_bar.pack_forget() # Hide initially

        # --- Canvas Bindings ---
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # --- Initial Setup ---
        self._update_ui_state() # Set initial button states

        # --- Center the window initially ---
        self.master.update_idletasks() 
        window_width = self.master.winfo_width()
        window_height = self.master.winfo_height()

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)

        self.master.geometry(f"+{center_x}+{center_y}")
        

    def _update_ui_state(self):
        """Updates the state of UI elements based on application state."""
        has_video = self.video_path is not None
        has_selected_points = len(self.selected_points_for_tracking) > 0
        has_tracked_data = self.tracked_points_cotracker is not None
        has_finalized_mask = bool(self.mask_polygons)

        # Select Points button logic
        if not has_video:
            self.select_points_button.config(state=tk.DISABLED, text="Select Points")
        elif has_tracked_data:
            self.select_points_button.config(state=tk.DISABLED, text="Select Points")
        else:
            self.select_points_button.config(state=tk.NORMAL, text="Select Points")

        # --- FIX: Consolidated and corrected logic for Export and Mask buttons ---
        # Determine the state of the export button first based on all conditions.
        export_state = tk.DISABLED
        if has_tracked_data and not self.is_masking_mode and has_finalized_mask:
            export_state = tk.NORMAL
        self.export_button.config(state=export_state)
        self.export_to_comfy_button.config(state=export_state)

        # Determine the state of the mask button.
        if not has_tracked_data:
            self.mask_button.config(state=tk.DISABLED, text="Draw Mask")
        elif self.is_masking_mode:
            self.mask_button.config(state=tk.NORMAL, text="Finalize Mask")
        else:
            self.mask_button.config(state=tk.NORMAL, text="Draw Mask")
        # --- End of Fix ---

        # Cursor logic (combined for all modes)
        if self.is_selecting_points:
            self.canvas.config(cursor="crosshair")
        elif self.is_masking_mode:
            if self.is_moving_mask_point:
                self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="tcross")
        else:
            self.canvas.config(cursor="")


        self.track_button.config(state=tk.NORMAL if has_video and has_selected_points and not has_tracked_data else tk.DISABLED)
        self.play_pause_button.config(state=tk.NORMAL if has_video else tk.DISABLED)
        self.slider.config(state=tk.NORMAL if has_video else tk.DISABLED)

        # Enable/Disable overlay checkboxes
        self.position_checkbox.config(state=tk.NORMAL if has_tracked_data else tk.DISABLED)
        can_calc_scale_rot = has_tracked_data and len(self.selected_points_for_tracking) == 2
        self.scale_checkbox.config(state=tk.NORMAL if can_calc_scale_rot else tk.DISABLED)
        self.rotation_checkbox.config(state=tk.NORMAL if can_calc_scale_rot else tk.DISABLED)

        if self.is_playing:
            self.play_pause_button.config(text="⏸") # Pause Icon
        else:
            self.play_pause_button.config(text="▶") # Play Icon

        # Disable all interactive elements during heavy processing
        is_processing = self.progress_bar.winfo_ismapped()
        self.load_video_button.config(state=tk.DISABLED if is_processing else tk.NORMAL)
        self.clear_button.config(state=tk.DISABLED if is_processing else tk.NORMAL)
        
        # Re-check some buttons that might be disabled by other logic even when not processing
        if is_processing:
            self.select_points_button.config(state=tk.DISABLED)
            self.track_button.config(state=tk.DISABLED)
            self.mask_button.config(state=tk.DISABLED)
            self.play_pause_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.DISABLED)
            self.export_to_comfy_button.config(state=tk.DISABLED)
            self.slider.config(state=tk.DISABLED)
            self.position_checkbox.config(state=tk.DISABLED)
            self.scale_checkbox.config(state=tk.DISABLED)
            self.rotation_checkbox.config(state=tk.DISABLED)
            

    def load_video(self):
        """Opens a file dialog to select a video and loads its first frame."""
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")])
        if not file_path:
            print("Video loading cancelled.")
            return

        self._full_clear_app_state(keep_video_dimensions=False) # Clear everything including old video dimensions
        self.video_path = file_path

        # Show progress bar for video loading
        self.progress_bar.pack(pady=5)
        # Set mode to determinate for video loading progress
        self.progress_bar.config(mode="determinate")
        self.update_progress_bar(0, "Loading video frames...")

        # Load video in a separate thread
        threading.Thread(target=self._load_video_in_thread).start()


    def _load_video_in_thread(self):
        """Loads video frames into a PyTorch tensor in a separate thread."""
        try:
            # Re-initialize OpenCV cap for reading frames
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                # ** MODIFIED: Pass a specific message **
                self.master.after(0, self._video_loading_finished, False, "Error: Could not open video file.")
                return

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # --- GPU-AWARE FAIL-SAFE ---
            estimated_memory_bytes = self.display_width * self.display_height * 3 * 4 * self.total_frames
            estimated_memory_gb = estimated_memory_bytes / (1024**3)
            print(f"Estimated video tensor memory usage: {estimated_memory_gb:.2f} GB.")

            limit_exceeded = False
            error_msg = ""

            if self.device == 'cuda':
                vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_limit_gb = vram_total_gb * 0.85
                print(f"GPU VRAM limit (85%): {vram_limit_gb:.2f} GB. Total VRAM: {vram_total_gb:.2f} GB.")
                
                if estimated_memory_gb > vram_limit_gb:
                    limit_exceeded = True
                    # ** This is the desired error message **
                    error_msg = (
                        f"Loading cancelled to prevent CUDA Out of Memory error. "
                        f"(Video: {estimated_memory_gb:.2f} GB, Limit: ~{vram_limit_gb:.1f} GB)"
                    )
            else:
                ram_total_gb = psutil.virtual_memory().total / (1024**3)
                ram_limit_gb = ram_total_gb * 0.75
                print(f"System RAM limit (75%): {ram_limit_gb:.2f} GB. Total RAM: {ram_total_gb:.2f} GB.")
                
                if estimated_memory_gb > ram_limit_gb:
                    limit_exceeded = True
                    error_msg = (
                        f"Loading cancelled to prevent system instability. "
                        f"(Video: {estimated_memory_gb:.2f} GB, Limit: ~{ram_limit_gb:.1f} GB)"
                    )

            if limit_exceeded:
                print(error_msg)
                # ** MODIFIED: Pass the detailed error_msg to the callback **
                self.master.after(0, self._video_loading_finished, False, error_msg)
                self.cap.release()
                return
            # --- END OF FAIL-SAFE ---

            if self.display_width == 0 or self.display_height == 0:
                self.master.after(0, self._video_loading_finished, False, "Error: Invalid video dimensions.")
                return

            # Read all frames into a list of NumPy arrays
            frames_list = []
            for i in range(self.total_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame)
                progress = int((i + 1) / self.total_frames * 100)
                self.master.after(0, lambda p=progress: self.update_progress_bar(p, f"Loading frames: {p}%"))

            self.cap.release()

            self.video_frames_tensor = torch.from_numpy(np.stack(frames_list)).permute(0, 3, 1, 2)[None].float().to(self.device)
            print(f"Video tensor loaded. Shape: {self.video_frames_tensor.shape}")

            if self.cotracker_model is None:
                self.master.after(0, lambda: self.update_progress_bar(0, "Loading CoTracker model..."))
                self.cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)
                self.master.after(0, lambda: self.update_progress_bar(100, "CoTracker model loaded."))

            self.master.after(0, self._video_loading_finished, True)

        except Exception as e:
            print(f"An error occurred during video loading: {e}")
            # ** MODIFIED: Pass the specific exception message **
            self.master.after(0, self._video_loading_finished, False, f"Error: {e}")
            

    def _video_loading_finished(self, success, message=""):
        """Callback to run on the main thread after video loading completes."""
        self.progress_bar.pack_forget() # Hide progress bar
        
        if success:
            print("Video loaded successfully.")
            # Set slider range
            self.slider.config(to=self.total_frames - 1)
            
            # Set the slider's value to 0
            self.slider.set(0)
            
            # Adjust Canvas and Slider to Video Dimensions
            self.canvas.config(width=self.display_width, height=self.display_height)
            self.slider.config(length=self.display_width)
            self.master.geometry("") # Let Tkinter re-calculate optimal window size
            self.master.update_idletasks() # Ensure GUI updates immediately

            # Recenter the window
            window_width = self.master.winfo_width()
            window_height = self.master.winfo_height()
            screen_width = self.master.winfo_screenwidth()
            screen_height = self.master.winfo_screenheight()
            center_x = int(screen_width / 2 - window_width / 2)
            center_y = int(screen_height / 2 - window_height / 2)
            self.master.geometry(f"+{center_x}+{center_y}")

            # Display the first frame
            self.display_frame(0)

            self.select_points_button.config(state=tk.NORMAL) # Enable point selection
            self.console_label.config(text=f"Video '{self.video_path}' loaded. Select points.")
        else:
            print("Video loading failed.")
            self.video_path = None # Clear video path on failure
            
            # ** THE FIX IS HERE: Display the specific message if provided **
            final_message = message if message else "Video loading failed. Please try again."
            self.console_label.config(text=final_message)

            # Reset canvas size to default if loading fails
            self.canvas.config(width=640, height=480)
            self.slider.config(length=640)
            self.master.geometry("")
            self.master.update_idletasks()
        
        self._update_ui_state() # Update button states after loading is done


    def display_frame(self, frame_index):
        """Displays a specific frame from the loaded video tensor."""
        if self.video_frames_tensor is None or frame_index < 0 or frame_index >= self.total_frames:
            return

        # Get frame from tensor (B T C H W -> C H W)
        frame_tensor = self.video_frames_tensor[0, frame_index].cpu().numpy()
        
        # Permute from (C, H, W) to (H, W, C) for PIL
        frame_numpy_hwc = np.transpose(frame_tensor, (1, 2, 0)) # Change to (H, W, C)
        
        # Convert RGB numpy array to PIL Image
        img = Image.fromarray(frame_numpy_hwc.astype(np.uint8))

        # Resize image to fit canvas, maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width == 0 or canvas_height == 0: # Fallback if canvas not yet rendered
            canvas_width = self.display_width
            canvas_height = self.display_height

        img_aspect = img.width / img.height
        canvas_aspect = canvas_width / canvas_height

        if img_aspect > canvas_aspect:
            new_width = canvas_width
            new_height = int(new_width / img_aspect)
        else:
            new_height = canvas_height
            new_width = int(new_height * img_aspect)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(canvas_width/2, canvas_height/2, image=self.photo, anchor=tk.CENTER)
        self.canvas.image = self.photo # Keep a reference!

        self.current_replay_index = frame_index
        
        # Temporarily disable slider command to prevent recursion
        self.slider.config(command="") # Disable the command
        self.slider.set(frame_index)
        self.slider.config(command=self.slider_moved) # Re-enable the command

        self._clear_canvas_overlays() # Clear all dynamic overlays before redrawing
        self._draw_selected_points_on_canvas() # Always draw manually selected points if any
        self._draw_tracked_points_on_canvas()
        self._draw_tracking_overlays() # Call new overlay drawing function
        self._draw_mask_on_canvas() # Draw mask overlays


    def slider_moved(self, val):
        """Callback for slider movement."""
        frame_index = int(float(val))
        self.display_frame(frame_index)
        # If playing, pause when slider is moved manually
        if self.is_playing:
            self.toggle_play_pause()


    def toggle_point_selection_mode(self):
        """Enters point selection mode; subsequent clicks do nothing."""
        # Do nothing if tracking data already exists
        if self.tracked_points_cotracker is not None:
            print("Cannot select new points while tracking data exists. Please 'Clear All' first.")
            return

        # Do nothing if already in point selection mode
        if self.is_selecting_points:
            self.slider.set(0)
            return

        # Activate point selection mode
        self.is_selecting_points = True
        
        print("Point selection mode active. Click on the video to select points.")
        self.selected_points_for_tracking = [] # Clear previous selections
        self.is_masking_mode = False # Turn off masking mode if active
        self.is_moving_mask_point = False # Ensure not in moving mode
        self.moving_point_index = -1
        self.canvas.unbind("<B1-Motion>") # Clear any previous bindings
        self.canvas.unbind("<ButtonRelease-1>")
        self._clear_canvas_overlays() # Clear old point drawings
        self.slider.set(0) 
        self.display_frame(0) # Redraw to show cleared points

        self._update_ui_state()


    def toggle_masking_mode(self):
        """Toggles the mask drawing mode."""
        if not (self.tracked_points_cotracker is not None): # Must have tracking data
            print("Masking mode requires successful tracking data first.")
            return

        self.is_masking_mode = not self.is_masking_mode

        if self.is_masking_mode:
            # Entering masking mode
            print("Masking mode active. Click on video to draw mask points. Click 'Finalize Mask' to finalize.")
            self.display_frame(0)
            
            # If a mask exists, make it editable. This allows re-editing.
            if self.mask_polygons:
                self.current_mask_polygon_points = list(self.mask_polygons.pop(0))
            else:
                # No existing masks, start a new one
                self.current_mask_polygon_points = []

            self.is_selecting_points = False # Turn off point selection mode if active
            self.is_moving_mask_point = False # Ensure not in moving mode
            self.moving_point_index = -1
            self.canvas.unbind("<B1-Motion>") # Clear any previous bindings
            self.canvas.unbind("<ButtonRelease-1>")
        else:
            # Exiting masking mode (Finalizing)
            if len(self.current_mask_polygon_points) > 2: # A polygon needs at least 3 points
                self.mask_polygons.append(list(self.current_mask_polygon_points)) # Add a copy
                
                # FIX: Always update the export mask with the latest finalized points.
                self.initial_mask_polygon_for_export = np.array(self.current_mask_polygon_points, dtype=float)
                
                print(f"Mask polygon finalized with {len(self.current_mask_polygon_points)} points.")
            elif len(self.current_mask_polygon_points) > 0:
                print(f"Current mask polygon had {len(self.current_mask_polygon_points)} points and was not added (needs >= 3 points).")
            
            self.current_mask_polygon_points = [] # Clear for next time
            
            self.is_moving_mask_point = False # Ensure not in moving mode
            self.moving_point_index = -1
            self.canvas.unbind("<B1-Motion>") # Clear bindings
            self.canvas.unbind("<ButtonRelease-1>")

        self._update_ui_state()
        self._clear_canvas_overlays() # Clear existing drawn mask
        self.display_frame(self.current_replay_index) # Redraw to update mask visibility


    def on_canvas_click(self, event):
        """Handles mouse clicks on the video canvas for point selection or mask drawing."""
        if self.is_selecting_points:
            if len(self.selected_points_for_tracking) >= 2:
                print("Maximum 2 points allowed for tracking.")
                return
            x_orig = int(event.x / self.canvas.winfo_width() * self.display_width)
            y_orig = int(event.y / self.canvas.winfo_height() * self.display_height)
            self.selected_points_for_tracking.append((x_orig, y_orig))
            print(f"Tracking point selected: ({x_orig}, {y_orig}). Total: {len(self.selected_points_for_tracking)}")
            
            # Automatically check scale and rotation if 2 points are selected
            if len(self.selected_points_for_tracking) == 2:
                self.show_scale.set(True)
                self.show_rotation.set(True)

            self._draw_selected_points_on_canvas() # Draw immediately
            self._update_ui_state() # Update track button state
        elif self.is_masking_mode:
            x_canvas_click = event.x
            y_canvas_click = event.y

            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Check if an existing point is clicked for moving
            clicked_on_existing_point_index = -1
            for i, (x_orig, y_orig) in enumerate(self.current_mask_polygon_points):
                x_canvas_point = x_orig * (canvas_width / self.display_width)
                y_canvas_point = y_orig * (canvas_height / self.display_height)
                
                distance = math.sqrt((x_canvas_click - x_canvas_point)**2 + (y_canvas_click - y_canvas_point)**2)
                if distance <= self.mask_point_snap_radius:
                    clicked_on_existing_point_index = i
                    break
            
            if clicked_on_existing_point_index != -1:
                # User clicked on an existing point, enter moving mode
                self.is_moving_mask_point = True
                self.moving_point_index = clicked_on_existing_point_index
                # Bind mouse motion and release events for dragging
                self.canvas.bind("<B1-Motion>", self.on_canvas_drag_mask_point)
                self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release_mask_point)
                self._update_ui_state() # Update cursor
            else:
                # No existing point clicked, add a new point
                x_orig = int(x_canvas_click / self.canvas.winfo_width() * self.display_width)
                y_orig = int(y_canvas_click / self.canvas.winfo_height() * self.display_height)
                self.current_mask_polygon_points.append((x_orig, y_orig))
                self._draw_mask_on_canvas() # Draw immediately
        else:
            print("Click 'Select Points' or 'Draw Mask' to enable interaction.")


    def on_canvas_drag_mask_point(self, event):
        """Handles dragging of a mask point on the canvas."""
        if self.is_masking_mode and self.is_moving_mask_point and self.moving_point_index != -1:
            x_orig = int(event.x / self.canvas.winfo_width() * self.display_width)
            y_orig = int(event.y / self.canvas.winfo_height() * self.display_height)
            
            # Update the specific point's coordinates
            self.current_mask_polygon_points[self.moving_point_index] = (x_orig, y_orig)
            self._draw_mask_on_canvas() # Redraw the mask to show the point moving

    def on_canvas_release_mask_point(self, event):
        """Handles releasing a dragged mask point on the canvas."""
        if self.is_masking_mode and self.is_moving_mask_point:
            self.is_moving_mask_point = False
            self.moving_point_index = -1
            # Unbind the drag and release events
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self._draw_mask_on_canvas() # Final redraw
            self._update_ui_state()


    def _draw_selected_points_on_canvas(self):
        """Draws the currently selected points on the canvas."""
        # Clear specific selected point tags, not all overlays
        self.canvas.delete("selected_point")
        self.canvas.delete("selected_point_text")

        # ONLY draw selected points if the application is in the 'selecting points' mode
        if not self.is_selecting_points:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        for i, (x_orig, y_orig) in enumerate(self.selected_points_for_tracking):
            x_canvas = x_orig * (canvas_width / self.display_width)
            y_canvas = y_orig * (canvas_height / self.display_height)
            
            # Draw circle
            radius = 5
            self.canvas.create_oval(x_canvas - radius, y_canvas - radius,
                                    x_canvas + radius, y_canvas + radius,
                                    fill="red", outline="white", width=1, tags="selected_point")
            # Draw point number
            self.canvas.create_text(x_canvas + 10, y_canvas, text=str(i + 1),
                                    fill="white", font=("Arial", 9, "bold"), tags="selected_point_text")
        self.canvas.tag_raise("selected_point")
        self.canvas.tag_raise("selected_point_text")


    def _clear_canvas_overlays(self):
        """Clears all dynamic overlays (points, tracks, and text overlays) from the canvas."""
        self.canvas.delete("selected_point")
        self.canvas.delete("selected_point_text")
        self.canvas.delete("tracked_point")
        self.canvas.delete("tracked_point_text")
        self.canvas.delete("tracked_line")
        self.canvas.delete("overlay_text") # New tag for general overlays
        self.canvas.delete("mask_overlay") # Clear mask overlays


    def start_tracking(self):
        """Initiates the CoTracker tracking process in a separate thread."""
        if self.video_frames_tensor is None:
            print("Please load a video first.")
            return
        if not self.selected_points_for_tracking:
            print("Please select points on the video first.")
            return

        print("Starting CoTracker tracking...")
        self.progress_bar.pack(pady=5)
        # Set mode to indeterminate and start animation to show activity
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start()
        self.update_progress_bar(0, "Tracking in progress... Please wait.") # Text message for user
        self._update_ui_state() # Disable buttons during tracking

        # Clear previous tracking results and calculated values
        self.tracked_points_cotracker = None
        self.tracked_visibility_cotracker = None
        self.current_display_point_coords = []
        self.current_display_point_visibility = []
        
        self.tracked_midpoints = None
        self.tracked_scales = None
        self.tracked_rotations = None

        self.initial_tracked_midpoint_val = None
        self.initial_tracked_distance_val = None
        self.initial_tracked_angle_val = None

        # Reset last valid transformation data for new tracking
        self.last_valid_midpoint = None
        self.last_valid_scale = 1.0
        self.last_valid_rotation_rad = 0.0

        # Clear mask tracking data
        self.initial_mask_centroid = None
        self.tracked_mask_centroid_coords = None


        # Start tracking in a new thread
        threading.Thread(target=self._run_cotracker_tracking).start()


    def _run_cotracker_tracking(self):
        """Performs CoTracker inference and calculates position, scale, rotation."""
        try:
            if self.cotracker_model == None:
                self.master.after(0, lambda: self.update_progress_bar(0, "Error: CoTracker model not loaded."))
                self.master.after(0, self._tracking_finished, False)
                return

            queries = []
            for x, y in self.selected_points_for_tracking:
                queries.append([0, x, y]) # Initial points are on frame 0

            queries_tensor = torch.tensor(queries).float().to(self.device)
            queries_tensor = queries_tensor[None] # Add batch dimension

            print(f"Running CoTracker on {self.total_frames} frames with {len(self.selected_points_for_tracking)} points...")
            
            with torch.no_grad():
                pred_tracks, pred_visibility = self.cotracker_model(
                    self.video_frames_tensor,
                    queries=queries_tensor
                )
            
            # --- Debugging: Check raw CoTracker output ---
            print(f"CoTracker raw output: pred_tracks shape: {pred_tracks.shape}, pred_visibility shape: {pred_visibility.shape}")
            if pred_tracks.numel() == 0 or pred_visibility.numel() == 0:
                print("Warning: CoTracker returned empty tensors.")
                self.master.after(0, self._tracking_finished, False)
                return

            self.tracked_points_cotracker = pred_tracks[0].cpu().numpy()
            self.tracked_visibility_cotracker = pred_visibility[0].cpu().numpy()
            print(f"After assignment: tracked_points_cotracker shape: {self.tracked_points_cotracker.shape}")
            print(f"After assignment: tracked_visibility_cotracker shape: {self.tracked_visibility_cotracker.shape}")
            # --- End Debugging ---

            print("CoTracker tracking complete. Calculating transformations...")

            num_points = len(self.selected_points_for_tracking)
            self.tracked_midpoints = np.zeros((self.total_frames, 2), dtype=float)
            
            initial_tracked_midpoint = None
            initial_tracked_distance = 0
            initial_tracked_angle = 0

            if num_points == 2:
                self.tracked_scales = np.ones(self.total_frames, dtype=float)
                self.tracked_rotations = np.zeros(self.total_frames, dtype=float)

                p1_init = np.array(self.selected_points_for_tracking[0])
                p2_init = np.array(self.selected_points_for_tracking[1])
                
                initial_tracked_distance = np.linalg.norm(p2_init - p1_init)
                if initial_tracked_distance == 0:
                    print("Warning: Initial points are identical. Scale and rotation will be undefined (initial_distance set to 1e-6).")
                    initial_tracked_distance = 1e-6 # Avoid division by zero, resulting in very large scale

                initial_tracked_angle = math.atan2(p2_init[1] - p1_init[1], p2_init[0] - p1_init[0])
                initial_tracked_midpoint = (p1_init + p2_init) / 2.0

                # Store initial values for mask transformation
                self.initial_tracked_midpoint_val = initial_tracked_midpoint
                self.initial_tracked_distance_val = initial_tracked_distance
                self.initial_tracked_angle_val = initial_tracked_angle


                for i in range(self.total_frames):
                    # Check visibility of both points
                    # Assuming a point is invisible if its score is <= 0.5
                    
                    # Get the visibility slice for the current frame
                    current_frame_visibility = self.tracked_visibility_cotracker[i]

                    # Adapt indexing based on the actual number of dimensions of the slice
                    if current_frame_visibility.ndim == 2: # Expected (N, 1)
                        p1_visible = current_frame_visibility[0, 0] > 0.5
                        p2_visible = current_frame_visibility[1, 0] > 0.5
                    elif current_frame_visibility.ndim == 1: # If last dimension was squeezed (N,)
                        p1_visible = current_frame_visibility[0] > 0.5
                        p2_visible = current_frame_visibility[1] > 0.5
                    else: # Fallback for unexpected shapes
                        print(f"ERROR: Unexpected visibility slice shape for frame {i}: {current_frame_visibility.shape}")
                        p1_visible = False
                        p2_visible = False


                    if p1_visible and p2_visible:
                        p1_curr = self.tracked_points_cotracker[i, 0]
                        p2_curr = self.tracked_points_cotracker[i, 1]

                        # Midpoint calculation
                        self.tracked_midpoints[i] = (p1_curr + p2_curr) / 2.0

                        # Scale calculation
                        current_distance = np.linalg.norm(p2_curr - p1_curr)
                        self.tracked_scales[i] = current_distance / initial_tracked_distance

                        # Rotation calculation
                        current_angle = math.atan2(p2_curr[1] - p1_curr[1], p2_curr[0] - p1_curr[0])
                        rotation_rad = current_angle - initial_tracked_angle
                        # Normalize angle to -180 to 180 degrees
                        rotation_deg = (rotation_rad + np.pi) % (2 * np.pi) - np.pi # Ensure it's between -pi and pi
                        rotation_deg = math.degrees(rotation_deg) # Convert to degrees
                        self.tracked_rotations[i] = rotation_deg
                    else:
                        # If one or both points are invisible, mark values as NaN
                        self.tracked_midpoints[i] = np.array([np.nan, np.nan])
                        self.tracked_scales[i] = np.nan
                        self.tracked_rotations[i] = np.nan

            elif num_points == 1:
                # If only one point, its position is the "midpoint" for display purposes
                self.tracked_midpoints = self.tracked_points_cotracker[:, 0, :]
                self.tracked_scales = None # Cannot calculate scale/rotation with one point
                self.tracked_rotations = None

                # Store initial values for mask transformation (for 1 point, midpoint is the point itself)
                # FIX: Ensure this is a float array to avoid integer-float mixing in mask transformations
                self.initial_tracked_midpoint_val = np.array(self.selected_points_for_tracking[0], dtype=float) # ADDED dtype=float
                self.initial_tracked_distance_val = 1.0 # Not applicable, set to default
                self.initial_tracked_angle_val = 0.0 # Not applicable, set to default
            else:
                self.tracked_midpoints = None
                self.tracked_scales = None
                self.tracked_rotations = None

                self.initial_tracked_midpoint_val = None
                self.initial_tracked_distance_val = None
                self.initial_tracked_angle_val = None


            print("Transformation calculations complete.")

            # --- NEW: Calculate and track mask centroid ---
            self.initial_mask_centroid = None
            self.tracked_mask_centroid_coords = None

            if self.mask_polygons and num_points >= 1: # Only process if a mask exists and we have at least one tracked point
                # For simplicity, track the first completed polygon
                target_mask_points = self.mask_polygons[0] 
                
                if target_mask_points:
                    # Calculate initial centroid of the first mask
                    sum_x = sum(p[0] for p in target_mask_points)
                    sum_y = sum(p[1] for p in target_mask_points)
                    mask_num_points = len(target_mask_points)
                    self.initial_mask_centroid = np.array([sum_x / mask_num_points, sum_y / mask_num_points], dtype=float)
                    print(f"Initial mask centroid calculated: {self.initial_mask_centroid}")

                    # Track the mask centroid through time
                    self.tracked_mask_centroid_coords = np.zeros((self.total_frames, 2), dtype=float)
                    
                    # Ensure initial_tracked_midpoint is valid for transformations
                    if self.initial_tracked_midpoint_val is None: # Should be set by now, but as a fallback
                         if num_points == 1:
                             self.initial_tracked_midpoint_val = np.array(self.selected_points_for_tracking[0])
                         else: # Should not happen if num_points is not 1 or 2
                             self.initial_tracked_midpoint_val = np.array([np.nan,np.nan]) # Default, will likely result in NaNs

                    for i in range(self.total_frames):
                        current_mask_centroid_untransformed = np.copy(self.initial_mask_centroid)
                        
                        # Check visibility of tracked points for the current frame
                        # For mask centroid, if the underlying tracking points are invisible, the mask centroid is also "invisible".
                        current_frame_visibility_scores = self.tracked_visibility_cotracker[i]
                        
                        any_tracked_point_visible = False
                        for score in current_frame_visibility_scores:
                            if score > 0.5: # Assuming >0.5 means visible
                                any_tracked_point_visible = True
                                break
                        
                        # FIX: Use .any() to check if either coordinate of the midpoint is NaN
                        if not any_tracked_point_visible or np.isnan(self.tracked_midpoints[i]).any():
                            self.tracked_mask_centroid_coords[i] = np.array([np.nan, np.nan])
                            continue # Skip if main tracking points are not visible or midpoint is NaN

                        current_tracked_midpoint = self.tracked_midpoints[i]
                        
                        # 1. Position of mask centroid relative to the initial pivot (tracked points' midpoint)
                        mask_centroid_relative_to_initial_pivot = current_mask_centroid_untransformed - self.initial_tracked_midpoint_val
                        
                        # Start with this relative position
                        transformed_relative_point = np.copy(mask_centroid_relative_to_initial_pivot)

                        # 2. Apply Scale (if enabled and only if 2 points are tracked)
                        if self.show_scale.get() and num_points == 2 and self.tracked_scales is not None and i < self.tracked_scales.shape[0]:
                            scale_factor = self.tracked_scales[i]
                            if not np.isnan(scale_factor):
                                transformed_relative_point *= scale_factor

                        # 3. Apply Rotation (if enabled and only if 2 points are tracked)
                        if self.show_rotation.get() and num_points == 2 and self.tracked_rotations is not None and i < self.tracked_rotations.shape[0]:
                            rotation_angle_deg = self.tracked_rotations[i]
                            if not np.isnan(rotation_angle_deg):
                                rotation_angle_rad = math.radians(rotation_angle_deg)
                                
                                # Perform 2D rotation
                                x_rel, y_rel = transformed_relative_point
                                rotated_x_rel = x_rel * math.cos(rotation_angle_rad) - y_rel * math.sin(rotation_angle_rad)
                                rotated_y_rel = x_rel * math.sin(rotation_angle_rad) + y_rel * math.cos(rotation_angle_rad)
                                transformed_relative_point = np.array([rotated_x_rel, rotated_y_rel])
                        
                        # 4. Apply Translation (by adding the current tracked midpoint)
                        self.tracked_mask_centroid_coords[i] = transformed_relative_point + current_tracked_midpoint
                    
                    print(f"Tracked mask centroid coords (first 5 frames): {self.tracked_mask_centroid_coords[:5]}")
                else:
                    print("No points found in the first mask polygon. Cannot calculate centroid.")
            # else:
                # print("No mask polygons defined or insufficient tracked points for mask tracking. Mask centroid will not be tracked.")
            # --- END NEW ---

            self.master.after(0, self._tracking_finished, True)

        except Exception as e:
            print(f"An error occurred during CoTracker tracking: {e}")
            # Ensure tracked data is cleared if an exception occurs
            self.tracked_points_cotracker = None
            self.tracked_visibility_cotracker = None
            self.tracked_midpoints = None
            self.tracked_scales = None
            self.tracked_rotations = None
            self.initial_tracked_midpoint_val = None # Clear initial tracked point values on error
            self.initial_tracked_distance_val = None # Clear initial tracked point values on error
            self.initial_tracked_angle_val = None # Clear initial tracked point values on error
            self.initial_mask_centroid = None # Clear mask tracking data on error
            self.tracked_mask_centroid_coords = None # Clear mask tracking data on error

            # Also clear last_valid_ values on error
            self.last_valid_midpoint = None
            self.last_valid_scale = 1.0
            self.last_valid_rotation_rad = 0.0

            self.master.after(0, lambda: self.update_progress_bar(0, f"Tracking error: {e}"))
            self.master.after(0, self._tracking_finished, False)


    def _tracking_finished(self, success):
        """Callback after tracking thread completes."""
        self.progress_bar.stop() # Stop indeterminate animation
        self.progress_bar.config(mode="determinate") # Reset mode to determinate for other uses
        self.progress_bar.pack_forget() # Hide progress bar
        
        # --- Debugging: Check state before updating UI ---
        print(f"Tracking finished. Success: {success}. self.tracked_points_cotracker is None: {self.tracked_points_cotracker is None}")
        # --- End Debugging ---

        if success:
            print("Tracking results available.")
            # Crucially, set is_selecting_points to False after successful tracking
            self.is_selecting_points = False 

            # Initialize last_valid_ values from frame 0 if valid
            # FIX: Use .any() to check if either coordinate of the midpoint is NaN
            if (self.tracked_midpoints is not None and self.tracked_midpoints.shape[0] > 0 and 
                not np.isnan(self.tracked_midpoints[0]).any()):
                self.last_valid_midpoint = np.array(self.tracked_midpoints[0])
                
                # Check for 2 points for scale/rotation
                if len(self.selected_points_for_tracking) == 2 and self.tracked_scales is not None and self.tracked_rotations is not None:
                    if not np.isnan(self.tracked_scales[0]) and not np.isnan(self.tracked_rotations[0]):
                        self.last_valid_scale = self.tracked_scales[0]
                        self.last_valid_rotation_rad = math.radians(self.tracked_rotations[0])
                        # Automatically check scale and rotation if 2 points are successfully tracked
                        self.show_scale.set(True)
                        self.show_rotation.set(True)
                    else: # If frame 0 scale/rot is NaN, use default
                        self.last_valid_scale = 1.0
                        self.last_valid_rotation_rad = 0.0
                else: # For 1 point, scale/rotation are default
                    self.last_valid_scale = 1.0
                    self.last_valid_rotation_rad = 0.0
            else: # If frame 0 is NaN or data is missing, reset last_valid_midpoint
                self.last_valid_midpoint = None
                self.last_valid_scale = 1.0
                self.last_valid_rotation_rad = 0.0

            self.display_frame(self.current_replay_index) # Redraw current frame with tracks and overlays
        else:
            self.tracked_points_cotracker = None
            self.tracked_visibility_cotracker = None
            self.tracked_midpoints = None
            self.tracked_scales = None
            self.tracked_rotations = None
            self.initial_tracked_midpoint_val = None
            self.initial_tracked_distance_val = None
            self.initial_tracked_angle_val = None
            self.initial_mask_centroid = None # Clear mask tracking data on error
            self.tracked_mask_centroid_coords = None # Clear mask tracking data on error

            # Also clear last_valid_ values on error
            self.last_valid_midpoint = None
            self.last_valid_scale = 1.0
            self.last_valid_rotation_rad = 0.0

            print("Tracking failed.")
            # If tracking failed, it might be desirable to remain in selection mode or revert to it
            self.is_selecting_points = False # Ensure selection mode is off even on failure

        self._update_ui_state() # Update button states after loading is done, including select points button


    def _draw_tracked_points_on_canvas(self):
        """Draws the tracked points on the current frame of the canvas."""
        # Note: _clear_canvas_overlays is now called by display_frame,
        # so it clears all dynamic overlays including this.
        self.canvas.delete("tracked_point")
        self.canvas.delete("tracked_point_text")

        if self.tracked_points_cotracker is None or self.tracked_visibility_cotracker is None:
            return

        if self.current_replay_index >= self.tracked_points_cotracker.shape[0]:
            return # No tracking data for this frame

        # Get tracked points and visibility for the current frame
        # (N 2) for points, (N 1) for visibility
        frame_points = self.tracked_points_cotracker[self.current_replay_index]
        frame_visibility = self.tracked_visibility_cotracker[self.current_replay_index]

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        for i in range(frame_points.shape[0]):
            x_orig, y_orig = frame_points[i]
            
            # Safely extract the scalar visibility score
            # Check current frame's visibility slice shape for this point
            if frame_visibility.ndim == 2: # Expected (N, 1)
                visibility_score = frame_visibility[i, 0]
            elif frame_visibility.ndim == 1: # If last dimension was squeezed (N,)
                visibility_score = frame_visibility[i]
            else: # Fallback
                visibility_score = 0.0 # Treat as invisible


            # Only draw if visible (you can set a threshold, e.g., > 0.5)
            if visibility_score > 0.5: # CoTracker outputs a score, typically 1.0 for visible
                x_canvas = x_orig * (canvas_width / self.display_width)
                y_canvas = y_orig * (canvas_height / self.display_height)

                radius = 3
                self.canvas.create_oval(x_canvas - radius, y_canvas - radius,
                                        x_canvas + radius, y_canvas + radius,
                                        fill="lime", outline="white", width=1, tags="tracked_point")
                # Optionally draw point number for tracked points
                self.canvas.create_text(x_canvas + 8, y_canvas, text=str(i + 1),
                                        fill="white", font=("Arial", 7), tags="tracked_point_text")
        self.canvas.tag_raise("tracked_point")
        self.canvas.tag_raise("tracked_point_text")


    def _draw_tracking_overlays(self):
        """Draws position, scale, and rotation overlays based on checkbox states."""
        self.canvas.delete("overlay_text") # Clear previous overlay text

        if self.tracked_points_cotracker is None: # No tracking data, nothing to overlay
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Overlay display position (top-left, adjust as needed)
        text_y_offset = 20
        line_height = 20

        # --- Position Overlay ---
        if self.show_position.get() and self.tracked_midpoints is not None:
            if self.current_replay_index < self.tracked_midpoints.shape[0]:
                mid_x, mid_y = self.tracked_midpoints[self.current_replay_index]
                if not np.isnan(mid_x) and not np.isnan(mid_y): # Check if position is valid (not NaN)
                    # Convert original video coords to canvas coords for display
                    mid_x_canvas = mid_x * (canvas_width / self.display_width)
                    mid_y_canvas = mid_y * (canvas_height / self.display_height)
                    text = f"Pos: ({int(mid_x_canvas)}, {int(mid_y_canvas)})"
                    self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="yellow", font=("Arial", 10), tags="overlay_text")
                else:
                    text = "Pos: N/A (Points Invisible)"
                    self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="gray", font=("Arial", 10), tags="overlay_text")
                text_y_offset += line_height
            
        # --- Scale Overlay ---
        if self.show_scale.get():
            if len(self.selected_points_for_tracking) == 2 and self.tracked_scales is not None:
                if self.current_replay_index < self.tracked_scales.shape[0]:
                    scale_val = self.tracked_scales[self.current_replay_index]
                    if not np.isnan(scale_val): # Check if scale is valid (not NaN)
                        text = f"Scale: {scale_val:.2f}x"
                        self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="yellow", font=("Arial", 10), tags="overlay_text")
                    else:
                        text = "Scale: N/A (Points Invisible)"
                        self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="gray", font=("Arial", 10), tags="overlay_text")
                text_y_offset += line_height
            else:
                text = "Scale: N/A (Requires 2 points)"
                self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="gray", font=("Arial", 10), tags="overlay_text")
                text_y_offset += line_height


        # --- Rotation Overlay ---
        if self.show_rotation.get():
            if len(self.selected_points_for_tracking) == 2 and self.tracked_rotations is not None:
                if self.current_replay_index < self.tracked_rotations.shape[0]:
                    rotation_val = self.tracked_rotations[self.current_replay_index]
                    if not np.isnan(rotation_val): # Check if rotation is valid (not NaN)
                        text = f"Rot: {rotation_val:.1f}°"
                        self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="yellow", font=("Arial", 10), tags="overlay_text")
                    else:
                        text = "Rot: N/A (Points Invisible)"
                        self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="gray", font=("Arial", 10), tags="overlay_text")
                text_y_offset += line_height
            else:
                text = "Rot: N/A (Requires 2 points)"
                self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="gray", font=("Arial", 10), tags="overlay_text")
                text_y_offset += line_height

        # --- NEW: Tracked Mask Centroid Overlay ---
        if self.tracked_mask_centroid_coords is not None and self.current_replay_index < self.tracked_mask_centroid_coords.shape[0]:
            mask_center_x_orig, mask_center_y_orig = self.tracked_mask_centroid_coords[self.current_replay_index]
            
            if not np.isnan(mask_center_x_orig) and not np.isnan(mask_center_y_orig):
                mask_center_x_canvas = mask_center_x_orig * (canvas_width / self.display_width)
                mask_center_y_canvas = mask_center_y_orig * (canvas_height / self.display_height)

                # Draw a point for the tracked mask centroid
                radius = 6
                self.canvas.create_oval(mask_center_x_canvas - radius, mask_center_y_canvas - radius,
                                        mask_center_x_canvas + radius, mask_center_y_canvas + radius,
                                        fill="cyan", outline="purple", width=2, tags="overlay_text") 

                # Display mask centroid coordinates
                text = f"Mask Centroid: ({int(mask_center_x_canvas)}, {int(mask_center_y_canvas)})"
                self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="cyan", font=("Arial", 10), tags="overlay_text")
            else:
                text = "Mask Centroid: N/A (Tracking points invisible)"
                self.canvas.create_text(10, text_y_offset, text=text, anchor="nw", fill="gray", font=("Arial", 10), tags="overlay_text")
            text_y_offset += line_height # Always advance offset even if N/A

        self.canvas.tag_raise("overlay_text")


    def _draw_mask_on_canvas(self):
        """Draws the current mask polygon(s) on the canvas, transforming completed masks."""
        self.canvas.delete("mask_overlay") # Clear all mask related items

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Helper function to draw a single polygon's points and lines
        def draw_polygon_points_and_lines(points_list, color, width, point_radius, line_tags, point_tags, current_moving_point_index=-1):
            canvas_points = []
            for i, (x_orig, y_orig) in enumerate(points_list):
                # Explicitly cast to int for Tkinter drawing functions
                x_canvas = int(x_orig * (canvas_width / self.display_width))
                y_canvas = int(y_orig * (canvas_height / self.display_height))
                canvas_points.append((x_canvas, y_canvas))

                # Draw point
                point_fill_color = color
                point_outline_color = "white"
                point_outline_width = 0.5
                point_draw_radius = point_radius

                if i == current_moving_point_index: # Highlight the point being moved
                    point_fill_color = "green" # Different color for the moving point
                    point_outline_color = "red" # Stronger outline
                    point_draw_radius = point_radius + 2 # Slightly larger
                    point_outline_width = 1.5 # Increased width for outline for highlight

                self.canvas.create_oval(x_canvas - point_draw_radius, y_canvas - point_draw_radius,
                                        x_canvas + point_draw_radius, y_canvas + point_draw_radius,
                                        fill=point_fill_color, outline=point_outline_color, width=point_outline_width, tags=point_tags)
            
            # Draw lines connecting points
            if len(canvas_points) > 1:
                for i in range(len(canvas_points) - 1):
                    p1_x, p1_y = canvas_points[i]
                    p2_x, p2_y = canvas_points[i+1]
                    self.canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill=color, width=width, tags=line_tags)
                
                if len(points_list) >= 3: # A polygon needs at least 3 points to be closed
                    p_last_x, p_last_y = canvas_points[-1]
                    p_first_x, p_first_y = canvas_points[0]
                    self.canvas.create_line(p_last_x, p_last_y, p_first_x, p_first_y, fill=color, width=width, tags=line_tags)

        # Draw completed polygons (these need to be transformed based on tracking)
        if self.mask_polygons and self.tracked_points_cotracker is not None and self.initial_tracked_midpoint_val is not None:
            # We will transform the first completed mask polygon
            initial_mask_polygon = np.array(self.mask_polygons[0]) # Convert to numpy for easier math
            initial_tracked_pivot_point = self.initial_tracked_midpoint_val # The fixed original pivot

            # Get current frame tracking data
            current_midpoint_data = None
            current_scale_data = np.nan
            current_rotation_data = np.nan
            any_tracked_point_visible_this_frame = False

            if self.current_replay_index < self.total_frames and self.current_replay_index < self.tracked_points_cotracker.shape[0]:
                current_frame_visibility_scores = self.tracked_visibility_cotracker[self.current_replay_index]
                for score in current_frame_visibility_scores:
                    if score > 0.5:
                        any_tracked_point_visible_this_frame = True
                        break

                if self.tracked_midpoints is not None and self.current_replay_index < self.tracked_midpoints.shape[0]:
                    current_midpoint_data = self.tracked_midpoints[self.current_replay_index]
                
                num_tracked_points = len(self.selected_points_for_tracking)
                if num_tracked_points == 2:
                    if self.tracked_scales is not None and self.current_replay_index < self.tracked_scales.shape[0]:
                        current_scale_data = self.tracked_scales[self.current_replay_index]
                    if self.tracked_rotations is not None and self.current_replay_index < self.tracked_rotations.shape[0]:
                        current_rotation_data = self.tracked_rotations[self.current_replay_index]

            # Determine if current frame has valid tracking data
            # FIX: Use .any() to check if either coordinate of the midpoint is NaN
            is_current_frame_data_valid = (
                any_tracked_point_visible_this_frame and
                current_midpoint_data is not None and
                not np.isnan(current_midpoint_data).any()
            )
            if len(self.selected_points_for_tracking) == 2:
                is_current_frame_data_valid = is_current_frame_data_valid and not np.isnan(current_scale_data) and not np.isnan(current_rotation_data)

            # Choose which transformation parameters to use (current or last valid)
            effective_midpoint = None
            effective_scale_factor = 1.0
            effective_rotation_angle_rad = 0.0

            if is_current_frame_data_valid:
                effective_midpoint = current_midpoint_data
                self.last_valid_midpoint = np.copy(current_midpoint_data) # Update last valid
                
                if len(self.selected_points_for_tracking) == 2:
                    effective_scale_factor = current_scale_data
                    self.last_valid_scale = current_scale_data
                    effective_rotation_angle_rad = math.radians(current_rotation_data)
                    self.last_valid_rotation_rad = math.radians(current_rotation_data)
                else: # 1 point: scale/rotation are default
                    self.last_valid_scale = 1.0
                    self.last_valid_rotation_rad = 0.0

            elif self.last_valid_midpoint is not None:
                # Use last known good data
                effective_midpoint = self.last_valid_midpoint
                effective_scale_factor = self.last_valid_scale
                effective_rotation_angle_rad = self.last_valid_rotation_rad
                print(f"Frame {self.current_replay_index}: Using last known good tracking data for mask.")
            else:
                # No valid tracking data available at all (first frames or tracking failed)
                # Do not draw the transformed mask.
                pass # Already initialized effective_midpoint to None, so skip drawing transformed mask

            # Proceed to draw transformed mask ONLY if effective_midpoint is valid
            if effective_midpoint is not None:
                transformed_polygon_points = []
                for point_orig in initial_mask_polygon:
                    # 1. Translate point so pivot (initial_tracked_pivot_point) is at origin
                    point_relative_to_pivot = point_orig - initial_tracked_pivot_point
                    
                    # Start with this relative position
                    transformed_relative_point = np.copy(point_relative_to_pivot)

                    # 2. Apply Scale (CONDITIONAL)
                    if self.show_scale.get():
                        transformed_relative_point *= effective_scale_factor

                    # 3. Apply Rotation (CONDITIONAL)
                    if self.show_rotation.get():
                        x_rel, y_rel = transformed_relative_point
                        rotated_x_rel = x_rel * math.cos(effective_rotation_angle_rad) - y_rel * math.sin(effective_rotation_angle_rad)
                        rotated_y_rel = x_rel * math.sin(effective_rotation_angle_rad) + y_rel * math.cos(effective_rotation_angle_rad)
                        transformed_relative_point = np.array([rotated_x_rel, rotated_y_rel])
                    
                    # 4. Translate back by current effective midpoint
                    transformed_point = transformed_relative_point + effective_midpoint
                    
                    # Explicitly convert to int before creating the tuple
                    transformed_polygon_points.append(tuple(transformed_point.astype(int)))
                    

                draw_polygon_points_and_lines(transformed_polygon_points, "blue", 2, 4, "mask_overlay", "mask_overlay", -1)
            else:
                # print(f"Frame {self.current_replay_index}: Skipping transformed mask drawing due to no valid tracking data.")
                pass # Do not draw anything if no effective midpoint is found

        # Draw currently drawing polygon (yellow, thinner, if in masking mode)
        # This one is always drawn from current_mask_polygon_points, not transformed
        if self.is_masking_mode and self.current_mask_polygon_points:
            draw_polygon_points_and_lines(self.current_mask_polygon_points, "yellow", 1, 3, "mask_overlay", "mask_overlay", 
                                          self.moving_point_index if self.is_moving_mask_point else -1)

        self.canvas.tag_raise("mask_overlay")


    def toggle_play_pause(self):
        """Toggles video playback."""
        if self.is_playing:
            self.is_playing = False
            if self.playback_job_id:
                self.master.after_cancel(self.playback_job_id)
                self.playback_job_id = None
        else:
            self.is_playing = True
            if self.current_replay_index >= self.total_frames: # Reset if at end
                self.current_replay_index = 0
            self._play_next_frame()
        self._update_ui_state()


    def _play_next_frame(self):
        """Plays the next frame in the sequence."""
        if not self.is_playing:
            return

        if self.current_replay_index < self.total_frames:
            self.display_frame(self.current_replay_index)
            self.current_replay_index += 1
            delay_ms = int(1000 / self.fps) if self.fps > 0 else 30
            self.playback_job_id = self.master.after(delay_ms, self._play_next_frame)
        else:
            self.is_playing = False
            self.current_replay_index = 0 # Reset for next play
            self.slider.set(0) # Reset slider
            self._update_ui_state()
            
            
    def handle_export_normal(self):
        """Sets the flag to False and then calls the export method."""
        self.save_to_temp = False
        self.export_mask_video() # Execute the export immediately

    def handle_export_to_comfy(self):
        """Sets the flag to True and then calls the export method."""
        self.save_to_temp = True
        self.export_mask_video() # Execute the export immediately        
            
            
    def export_mask_video(self):
        """Prompts user for save location and starts mask video export in a new thread."""
        if not self.mask_polygons and self.initial_mask_polygon_for_export is None:
            print("No mask has been drawn and finalized to export.")
            return
            
        # FIX: Explicitly check for None instead of relying on truthiness of the array.
        if self.tracked_points_cotracker is None:
            print("No tracking data available to export mask video.")
            return
            
        if self.initial_mask_polygon_for_export is None:
            print("Initial mask polygon not set for export. Please re-draw and finalize a mask.")
            return

        if self.save_to_temp:
            file_path = os.path.join(self.output_path, self.output_filename)
            print(f"Exporting mask video to ComfyUI temp: {file_path}")
        else:         
            file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            title="Save Mask Video As"
            )
        if not file_path:
            print("Mask video export cancelled.")
            return

        self.progress_bar.pack(pady=5)
        self.progress_bar.config(mode="determinate")
        self.update_progress_bar(0, "Exporting mask video... Please wait.")
        self._update_ui_state() # Disable buttons

        threading.Thread(target=self._generate_and_save_mask_video_in_thread, args=(file_path,)).start()


    def _generate_and_save_mask_video_in_thread(self, output_filepath):
        """Generates and saves the mask video by piping frames to an FFmpeg subprocess."""
        try:
            print(f"Starting mask video export to {output_filepath} via FFmpeg subprocess...")

            # 1. Setup video parameters
            valid_fps = self.fps if self.fps > 0 else 30
            if self.fps <= 0:
                print(f"Warning: Invalid source FPS ({self.fps}). Defaulting to {valid_fps} FPS for export.")

            # 2. Construct the FFmpeg command
            command = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                
                # Input stream parameters
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',  # The format of the numpy array data
                '-s', f'{self.display_width}x{self.display_height}',
                '-r', str(valid_fps),
                '-i', '-',  # Read input from stdin (the pipe)

                # Output stream parameters
                '-c:v', 'libx264',  # H.264 codec for compatibility
                '-pix_fmt', 'yuv420p',  # Standard pixel format for web video
                '-crf', '23',  # A good quality/size balance (lower is better quality)
                '-preset', 'medium', # A balance between encoding speed and compression
                output_filepath
            ]

            # 3. Open the subprocess with a pipe for stdin
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 4. Loop through frames and write them to the pipe
            for i in range(self.total_frames):
                progress = int((i + 1) / self.total_frames * 100)
                self.master.after(0, lambda p=progress: self.update_progress_bar(p, f"Exporting frame {i+1}/{self.total_frames}: {p}%"))

                mask_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                effective_midpoint, effective_scale_factor, effective_rotation_angle_rad = self._get_mask_transformation_for_frame(i)

                if effective_midpoint is not None:
                    # (The logic to calculate transformed_polygon_points remains the same)
                    transformed_polygon_points = []
                    initial_tracked_pivot_point = self.initial_tracked_midpoint_val
                    for point_orig in self.initial_mask_polygon_for_export:
                        point_relative_to_pivot = point_orig - initial_tracked_pivot_point
                        transformed_relative_point = np.copy(point_relative_to_pivot)
                        if self.show_scale.get():
                            transformed_relative_point *= effective_scale_factor
                        if self.show_rotation.get():
                            x_rel, y_rel = transformed_relative_point
                            rotated_x_rel = x_rel * math.cos(effective_rotation_angle_rad) - y_rel * math.sin(effective_rotation_angle_rad)
                            rotated_y_rel = x_rel * math.sin(effective_rotation_angle_rad) + y_rel * math.cos(effective_rotation_angle_rad)
                            transformed_relative_point = np.array([rotated_x_rel, rotated_y_rel])
                        transformed_point = transformed_relative_point + effective_midpoint
                        transformed_polygon_points.append(transformed_point)
                    
                    polygon_pts_int = np.array(transformed_polygon_points, dtype=np.int32)
                    cv2.fillPoly(mask_frame, [polygon_pts_int], (255, 255, 255), lineType=cv2.LINE_AA)
                
                # Write the raw pixel data to the FFmpeg process
                proc.stdin.write(mask_frame.tobytes())

            # 5. Finalize the video
            stdout, stderr = proc.communicate() # Close stdin and wait for the process to finish
            if proc.returncode != 0:
                # If FFmpeg returns an error, raise an exception with its output
                raise RuntimeError(f"FFmpeg error:\n{stderr.decode('utf-8')}")

            self.master.after(0, self._export_finished, True)
            print("Mask video export complete.")
            
            if self.save_to_temp:
                self.master.withdraw()
                self.quit_app()

        except Exception as e:
            print(f"An error occurred during mask video export: {e}")
            self.master.after(0, lambda err=e: self.update_progress_bar(0, f"Export error: {err}"))
            self.master.after(0, self._export_finished, False)

    def _get_mask_transformation_for_frame(self, frame_index):
        # This is a helper function to avoid code duplication. It's not in the original file, but cleans it up.
        # It determines the effective transformation parameters for a given frame.
        if self.tracked_points_cotracker is None or self.initial_tracked_midpoint_val is None:
            return None, 1.0, 0.0

        current_midpoint_data = None
        current_scale_data = np.nan
        current_rotation_data = np.nan
        any_tracked_point_visible_this_frame = False

        if frame_index < self.total_frames and frame_index < self.tracked_points_cotracker.shape[0]:
            current_frame_visibility_scores = self.tracked_visibility_cotracker[frame_index]
            for score in current_frame_visibility_scores:
                if score > 0.5:
                    any_tracked_point_visible_this_frame = True
                    break

            if self.tracked_midpoints is not None and frame_index < self.tracked_midpoints.shape[0]:
                current_midpoint_data = self.tracked_midpoints[frame_index]
            
            num_tracked_points = len(self.selected_points_for_tracking)
            if num_tracked_points == 2:
                if self.tracked_scales is not None and frame_index < self.tracked_scales.shape[0]:
                    current_scale_data = self.tracked_scales[frame_index]
                if self.tracked_rotations is not None and frame_index < self.tracked_rotations.shape[0]:
                    current_rotation_data = self.tracked_rotations[frame_index]

        is_current_frame_data_valid = (
            any_tracked_point_visible_this_frame and
            current_midpoint_data is not None and
            not np.isnan(current_midpoint_data).any()
        )
        if len(self.selected_points_for_tracking) == 2:
            is_current_frame_data_valid = is_current_frame_data_valid and not np.isnan(current_scale_data) and not np.isnan(current_rotation_data)

        if is_current_frame_data_valid:
            self.last_valid_midpoint = np.copy(current_midpoint_data)
            effective_midpoint = current_midpoint_data
            
            if len(self.selected_points_for_tracking) == 2:
                self.last_valid_scale = current_scale_data
                self.last_valid_rotation_rad = math.radians(current_rotation_data)
                effective_scale_factor = current_scale_data
                effective_rotation_angle_rad = math.radians(current_rotation_data)
            else:
                self.last_valid_scale = 1.0
                self.last_valid_rotation_rad = 0.0
                effective_scale_factor = 1.0
                effective_rotation_angle_rad = 0.0

            return effective_midpoint, effective_scale_factor, effective_rotation_angle_rad

        elif self.last_valid_midpoint is not None:
            return self.last_valid_midpoint, self.last_valid_scale, self.last_valid_rotation_rad
            
        else:
            return None, 1.0, 0.0

    def _export_finished(self, success):
        """Callback to run on the main thread after export completes."""
        self.progress_bar.stop()
        self.progress_bar.config(mode="determinate")
        self.progress_bar.pack_forget()
        if success:
            print(f"Mask video exported successfully.")
        else:
            print(f"Mask video export failed.")
        self._update_ui_state() # Re-enable buttons
        

    def update_progress_bar(self, value, text=""):
        """Updates the progress bar and console label."""
        self.progress_bar.config(value=value)
        self.console_label.config(text=text)
        self.master.update_idletasks() # Update GUI immediately


    def clear_tracking_and_points(self):
        """Clears only selected points, tracking data, and overlays, but keeps the video loaded."""
        self.is_playing = False
        if self.playback_job_id:
            self.master.after_cancel(self.playback_job_id)
            self.playback_job_id = None

        self.is_selecting_points = False
        self.is_masking_mode = False # Clear masking mode
        self.is_moving_mask_point = False # Clear moving point mode
        self.moving_point_index = -1
        self.canvas.unbind("<B1-Motion>") # Unbind potential drag events
        self.canvas.unbind("<ButtonRelease-1>")


        self.selected_points_for_tracking = []
        self.tracked_points_cotracker = None
        self.tracked_visibility_cotracker = None
        
        # Clear calculated tracking data
        self.tracked_midpoints = None
        self.tracked_scales = None
        self.tracked_rotations = None

        self.initial_tracked_midpoint_val = None
        self.initial_tracked_distance_val = None
        self.initial_tracked_angle_val = None

        # Reset last valid transformation data
        self.last_valid_midpoint = None
        self.last_valid_scale = 1.0
        self.last_valid_rotation_rad = 0.0

        self.current_display_point_coords = []
        self.current_display_point_visibility = []
        self.mask_polygons = [] # Clear saved mask polygons
        self.current_mask_polygon_points = [] # Clear current drawing mask points
        self.initial_mask_polygon_for_export = None # Clear export mask

        self.initial_mask_centroid = None # Clear mask tracking data
        self.tracked_mask_centroid_coords = None # Clear mask tracking data
        
        self._clear_canvas_overlays() # Clear only dynamic overlays
        
        # This unchecks the boxes by setting their control variables to False.
        self.show_position.set(True)
        self.show_scale.set(False)
        self.show_rotation.set(False)
        
        # Redraw the current frame to show video without points/overlays
        # Only if a video is actually loaded
        if self.video_path is not None:
            self.display_frame(0) 
            self.console_label.config(text=f"Points and tracking data cleared. Video '{self.video_path}' still loaded.")
        else:
            # If no video is loaded, perform a full clear state
            self._full_clear_app_state(keep_video_dimensions=False)
            self.console_label.config(text="All data cleared. Load a video to begin.")


        self._update_ui_state()
        print("Points and tracking data cleared.")


    def _full_clear_app_state(self, keep_video_dimensions=False):
        """Performs a full clear of all application state, including video resources.
        If keep_video_dimensions is True, it doesn't reset canvas/slider sizes.
        This is an internal helper called by load_video (false) and quit_app (true).
        """
        self.is_playing = False
        if self.playback_job_id:
            self.master.after_cancel(self.playback_job_id)
            self.playback_job_id = None

        self.is_selecting_points = False
        self.is_masking_mode = False # Clear masking mode
        self.is_moving_mask_point = False # Clear moving point mode
        self.moving_point_index = -1
        self.canvas.unbind("<B1-Motion>") # Unbind potential drag events
        self.canvas.unbind("<ButtonRelease-1>")

        self.selected_points_for_tracking = []
        self.tracked_points_cotracker = None
        self.tracked_visibility_cotracker = None
        
        # Clear calculated tracking data
        self.tracked_midpoints = None
        self.tracked_scales = None
        self.tracked_rotations = None

        self.initial_tracked_midpoint_val = None
        self.initial_tracked_distance_val = None
        self.initial_tracked_angle_val = None

        # Reset last valid transformation data
        self.last_valid_midpoint = None
        self.last_valid_scale = 1.0
        self.last_valid_rotation_rad = 0.0

        self.current_display_point_coords = []
        self.current_display_point_visibility = []
        self.mask_polygons = [] # Clear saved mask polygons
        self.current_mask_polygon_points = [] # Clear current drawing mask points
        self.initial_mask_polygon_for_export = None # Clear export mask

        self.initial_mask_centroid = None # Clear mask tracking data
        self.tracked_mask_centroid_coords = None # Clear mask tracking data
        
        self._clear_canvas_overlays()
        self.canvas.delete("all") # Clear all canvas items including image
        
        # This unchecks the boxes by setting their control variables to False.
        self.show_position.set(True)
        self.show_scale.set(False)
        self.show_rotation.set(False)

        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.video_frames_tensor = None
        self.total_frames = 0
        self.fps = 0
        self.video_path = None # Clear video path during full clear

        if not keep_video_dimensions:
            self.canvas.config(width=640, height=480) # Reset canvas size
            self.slider.config(length=640) # Reset slider length
            self.master.geometry("") # Let Tkinter reset window size
        
        self.slider.config(to=0) # Reset slider range to 0
        self.slider.set(0)
        self.console_label.config(text="Ready.")
        self.master.update_idletasks() # Ensure GUI updates immediately

        # Recenter the window after clearing data and resetting dimensions
        window_width = self.master.winfo_width()
        window_height = self.master.winfo_height()
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.master.geometry(f"+{center_x}+{center_y}")
            
        self._update_ui_state()
        print("Full application state cleared.")


    def quit_app(self):
        """Cleans up and exits the application."""
        self._full_clear_app_state(keep_video_dimensions=False) # Perform a full clear
        sys.stdout = self.console_redirector.stdout # Restore original stdout
        self.master.destroy()

# --- Main execution block ---
if __name__ == "__main__":
    root = tk.Tk()
    root.attributes('-topmost', True)
    app = PointTrackerApp(root)
    root.mainloop()