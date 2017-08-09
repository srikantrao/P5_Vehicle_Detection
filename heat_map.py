import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.ndimage.measurements import label

class heat_map:

    def __init__(self):
        self.heatmap = np.zeros((720,1280))
        self.threshold = 2
        self.labels = None
        self.frame_number = 0
        self.box_list = []
        self.boxes_tracked = []

    def add_heat(self,bbox_list):
    #Iterate through the list of bboxes

        for box in bbox_list:
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] +=1



    # Thresholding to reject false positives
    def apply_threshold(self):

        # Zero out the pixels using the threshold as a mask
        self.heatmap[self.heatmap < self.threshold] = 0

    def find_labels(self):
        self.labels = label(self.heatmap)



    def draw_labeled_bboxes(self):
        bbox = []
        for car_number in range(1,self.labels[1]+1):
            #Find pixels with each car number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))
        return bbox

    def check_bbox(self,bbox_list):
        updated_bbox = []
        #Update the heat map
        self.add_heat(bbox_list)
        # Apply the threshold to the bbox
        self.apply_threshold()
        # Find the labels
        self.find_labels()
        # find the final bbox
        updated_bbox = self.draw_labeled_bboxes()
        self.frame_number +=1
        self.heatmap = np.zeros((720, 1280))
        return updated_bbox






















