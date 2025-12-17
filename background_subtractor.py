import cv2
import numpy as np

class BackgroundDetector:
    def __init__(self):
        # MOG2 is a popular Background Subtraction algorithm.
        # history=500: It remembers the last 500 frames to build the model.
        # varThreshold=16: Threshold for "Mahalanobis distance" to decide if pixels are background.
        # detectShadows=False: Shadows can be tricky, let's keep it simple (binary).
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.learning_rate = -1 # -1 means auto learning rate
        
    def apply(self, frame):
        """
        Returns a binary mask where 255 = Moving Object, 0 = Background.
        """
        # Apply the subtractor to get the foreground mask
        fg_mask = self.subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Clean up the mask with morphology (remove small noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        return fg_mask
