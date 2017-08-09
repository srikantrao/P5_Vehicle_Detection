# Import everything needed to edit video clips
from moviepy.editor import *

clip = VideoFileClip("project_video_output.mp4").subclip(8,10)

video = CompositeVideoClip([clip])

video.write_videofile("project_video_8_10.mp4",audio=False)