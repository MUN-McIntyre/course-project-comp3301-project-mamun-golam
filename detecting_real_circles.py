import sys
import time
import numpy as np
import os
import cv2

sys.path.append('Hough_Circle_Detection')
sys.path.append('Circle-Detection-CNN')
from canny_circle_detection import find_circle_hough
from validation import find_circle_cnn, iou
from create_circles import draw_circle


def find_real_circle(dir):
    """
    Loops through the images in a directory and detects and draws circle using CNN and Hough Circles, and
    save the images with the detected circle drawn in the same directory.

    Args:
        dir (str): A string representing the directory that contains the image.
    """
    
    for filename in os.listdir('real_test'):
        filepath = os.path.join('real_test', filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        find_circle_hough(img, filename.split('.')[0], dir)
        find_circle_cnn(img, filename.split('.')[0], dir)

def main():
    find_real_circle('real_test')

if __name__ == '__main__':
    main()
