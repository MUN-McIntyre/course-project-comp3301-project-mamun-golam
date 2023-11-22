import cv2
import numpy as np
import random
import sys

sys.path.append('Circle-Detection-CNN')
from dataset import add_salt_and_pepper_noise


def draw_circle(filename):
    """
    Draw a circle on a 200x200 grayscale image using CV2 and add random noise and blurriness to the image.

    Args:
        i (str): A string representing the name of the image.

    Returns:
        list: the center and radius of the circle (col, row, rad), img and the img.
    """

    img = np.full((200,200), 255, dtype=np.uint8)

    # Create random center and radius of the circle
    center = (np.random.randint(10, 190), np.random.randint(10, 190))
    radius = np.random.randint(10, 100)

    # Create a create using cv2
    cv2.circle(img, center, radius, color=0, thickness=random.randint(1, 5))

    # Add noise with a random probability
    if np.random.randint(1, 3) == 1:
        img = add_salt_and_pepper_noise(img)

    # Add blur with a random probability and random kernel upto 5x5
    if np.random.randint(1, 3) == 1:    
        kernel_size = random.randrange(1, 6, 2)
        img = cv2.blur(img, ksize=(kernel_size, kernel_size))

    cv2.imwrite(f'test/{filename}.jpg', img)

    return (center[0], center[1], radius), img