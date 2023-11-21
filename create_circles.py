import cv2
import numpy as np
import random
import sys

sys.path.append('Circle-Detection-CNN')
from dataset import add_salt_and_pepper_noise

def draw_circle(i):

    img = np.full((200,200), 255, dtype=np.uint8)
    center = (np.random.randint(10, 190), np.random.randint(10, 190))
    radius = np.random.randint(10, 100)
    cv2.circle(img, center, radius, color=0, thickness=random.randint(1, 5))

    if np.random.randint(1, 3) == 1:
        img = add_salt_and_pepper_noise(img)

    if np.random.randint(1, 3) == 1:    
        kernel_size = random.randrange(1, 6, 2)
        img = cv2.blur(img, ksize=(kernel_size, kernel_size))

    cv2.imwrite(f'test/{i}.jpg', img)
    return (center[0], center[1], radius), img