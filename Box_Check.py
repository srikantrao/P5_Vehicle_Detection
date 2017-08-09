
import numpy as np
import cv2


class Box_Check:
    def __init__(self):
        self.box_list = []
        self.box_tracked = []

    def draw_bbox(self, img):
        """Copy of the Udacity function using the cv2 method to plot the boxes detected as actual objects/cars
        Draw the image only if it has been tracked for 5 more images than it has been lost"""

        imcopy = np.copy(img)

        for bbox in self.box_tracked:
            bb = bbox[0]
            if bbox[2] - bbox[3]> 5:
                cv2.rectangle(imcopy,bb[0],bb[1],(0,0,255),6)
        return imcopy

    def centroid(self, bbox):
        """ Find the centroid of a box"""
        return (int((bbox[1][0] - bbox[0][0]) / 2) + bbox[0][0], int((bbox[1][1] - bbox[0][1]) / 2) + bbox[0][1])

    def same_box(self, old_centroid, new_centroid):
        """Compare two centroid and see if they are a value k away from each other
        Might want to provide k as a parameter ... currently hardcoded to 32"""
        old_x = old_centroid[0]
        old_y = old_centroid[1]
        new_x = new_centroid[0]
        new_y = new_centroid[1]

        if new_x > old_x - 32 and new_x < old_x + 32 and new_y > old_y - 32 and new_y < old_y + 32:
            return True
        else:
            return False

    def match_box_with_tracked(self, box_centroid):
        """Track boxes detected with already detected boxes based on centroid distance"""
        match = False
        matched_box = []
        for box in self.box_list:
            new_centroid = box[1]
            box_tracked = box[2]
            if self.same_box(box_centroid, new_centroid) and box_tracked >= 1:
                match = True
                matched_box = box
                break

        return match, matched_box

    def track_cars(self):
        """Track the cars that are currently objects
        Updates the values that are going to be used by other functions
        Increment the track value and Decrement the lost value if it has been found again
        Increment the lost value if it has not been found
        Remove the car if it has reached threshold of 5 images
        """
        new_cars_tracked = []

        if len(self.box_tracked) == 0:
            if len(self.box_list) > 0:
                self.box_tracked = self.box_list
        else:
            # Iterate through tracked cars to find matches
            for car in self.box_tracked:
                car_bbox = car[0]
                box_centroid = car[1]
                car_tracked = car[2]
                car_lost = car[3]

                # Match the cars in new frame with tracked cars
                match, matched_box = self.match_box_with_tracked(box_centroid)
                if match:
                    new_cars_tracked.append((matched_box[0], matched_box[1], car_tracked + 1, max(0,car_lost-1)))
                    # Use tracked value of match car in car_list to indicate that we've found a match
                    index = self.box_list.index(matched_box)
                    # self.box_list[index][3] = 0
                    this_car = list(self.box_list[index])
                    this_car[2] = 0
                    self.box_list[index] = this_car
                else:
                    new_cars_tracked.append((car_bbox, box_centroid, car_tracked, car_lost + 1))

            # Remove tracked cars that have been lost for 5 frames or more
            for car in new_cars_tracked:
                car_tracked = car[2]
                car_lost = car[3]
                if car_lost >= 5 or car_lost >= car_tracked:
                    new_cars_tracked.remove(car)

            # Add any new cars that are not already included
            for car in self.box_list:
                car_bbox = car[0]
                box_centroid = car[1]
                car_tracked = car[2]
                if car_tracked == 1:
                    new_cars_tracked.append((car_bbox, box_centroid, 1, 0))

            self.box_tracked = new_cars_tracked

    def merge_overlaps(self, box1):
        """ This function is used to see if a box/car that is being tracked overlaps with this current box
        If there is overlap, then the top_left and bottom_right co-ordinates are changed to make it a union of the
        two boxes.
        This is required to reduce jittering of images as the boxes tracking the same object keep shifting"""
        merged = False
        # Check to see if box1 overlaps any box in boxes
        for existing_car in self.box_list:
            b1x1 = box1[0][0]
            b1y1 = box1[0][1]
            b1x2 = box1[1][0]
            b1y2 = box1[1][1]
            b2x1 = existing_car[0][0][0]
            b2y1 = existing_car[0][0][1]
            b2x2 = existing_car[0][1][0]
            b2y2 = existing_car[0][1][1]

            if (b2x1 >= b1x1 and b2x1 <= b1x2) or (b2x2 >= b1x1 and b2x2 <= b1x2) or (b2x1 <= b1x1 and b2x2 >= b1x2):
                if (b2y1 >= b1y1 and b2y1 <= b1y2) or (b2y2 >= b1y1 and b2y2 <= b1y2) or (
                        b2y1 <= b1y1 and b2y2 >= b1y2):

                    new_box = ((min(b1x1, b2x1), min(b1y1, b2y1)), (max(b1x2, b2x2), max(b1y2, b2y2)))
                    self.box_list.remove(existing_car)
                    self.box_list.append((new_box, self.centroid(new_box), 1, 0))
                    merged = True
                    break

        # If box has not been merged, then add it as a new box
        if merged == False:
            self.box_list.append((box1, self.centroid(box1), 1, 0))

    def detect(self, windows):
        """ Top level function called in the pipeline
        windows - Output of the heatmap
        Loop through the box list to track actual objects"""
        self.box_list = []
        # Loop through the list and merge overlaps
        for win in windows:
            bbox = win

            # First box always passes through
            if len(self.box_list) == 0:
                self.box_list.append((bbox, self.centroid(bbox), 1, 0))
                continue

            # Resolve overlaps of this box with final_bboxes
            self.merge_overlaps(bbox)

        # Track cars between frame
        self.track_cars()