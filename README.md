# About

This custom node pack (in development) provides a suite of tools for compositing and visual effects workflows within ComfyUI. 
Most notably, it features a robust point tracker powered by Meta's CoTracker model, alongside other nodes 
designed to streamline masking, tracking, and compositing tasks.

# Point Tracker

The Point Tracker functions similarly to tracking tools found in professional software like After Effects and Blender.
By tracking up to two points, it can calculate and generate position, rotation, and scale data. After generating a successful track
you can then create a custom mask polygon that can be used in your workflow.

![image](https://github.com/tnil25/ComfyUI-TJNodes/blob/master/images/pt_demo-ezgif.gif)

## User Guide

1. Drop the Point Tracker node on to the canvas.
2. Click 'Track Video' to open the Point Tracker GUI.
3. Click 'Load Video' and locate the video file you wish to track.
4. Select up to two points in the video to track. One point will only track position data, two points will track rotation and
   scale data as well.
5. Click 'Track Points' the CoTracker model will automatically be downloaded and tracking will begin.
6. Once the track is finished check it by playing back the video. If you aren't happy with it click 'Clear All' to restart.
7. Click 'Draw Mask' to draw a custom polygon on the video. Click 'Finalize Mask' to apply the tracking data to it.
   You can edit the mask by clicking 'Draw Mask' again.
8. Click 'Export to ComfyUI' to send the tracked mask back to ComfyUI, it will automatically be updated in the node.
9. You can also use 'Export As..." to save the mask to any location on your computer.

## Tips

* Point tracking works best when tracking contrasting areas of an image (eg. a button on a shirt, the corner of a window, a letter on a keyboard, etc.)
  It will almost always fail if you attempt to track a flat color (eg. the sky, a flat wall, etc.)
* In order to track scale and rotation two points are needed to calculate the tracking data, the farther away these points are the better,
  but make sure they are on the same plane of movement. Tracking will be inaccurate if for example one point is on the foreground and one on the background.
