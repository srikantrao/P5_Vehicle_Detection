from lesson_functions import *
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import glob
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from heat_map import heat_map
from moviepy.editor import VideoFileClip
from Box_Check import Box_Check

# Define the pipeline for the image

def image_pipeline(img):

    windows = full_sliding_window_search(img, clf, scaler)

    updated_windows = hm.check_bbox(windows)

    boxcheck.detect(updated_windows)

    window_img = boxcheck.draw_bbox(img)

    return window_img

# Load the classifier from the pickle file
svc = joblib.load('svc.pickle')
scaler = joblib.load('scaler.pickle')
clf = joblib.load('calibrated_classifier.pickle')
print("The classifier  and the scaler have been loaded successfully ..")

# Create a heat_map object
hm = heat_map()
boxcheck = Box_Check()
clip = VideoFileClip("./test_video.mp4")
output_video = "./test_video_output.mp4"
output_clip = clip.fl_image(image_pipeline)
output_clip.write_videofile(output_video, audio=False)

