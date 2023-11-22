import numpy as np
from skimage.draw import circle_perimeter_aa
import csv
import random
import cv2


def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    row, col = image.shape
    noisy = image.copy()

    # Add salt noise
    salt_pixels = np.random.choice((0, 1), size=(row, col), p=[1 - salt_prob, salt_prob])
    noisy[salt_pixels == 1] = 255

    # Add pepper noise
    pepper_pixels = np.random.choice((0, 1), size=(row, col), p=[1 - pepper_prob, pepper_prob])
    noisy[pepper_pixels == 1] = 0

    return noisy.astype(np.uint8)


def noisy_circle(size, radius):
    # CHANGED TO FLOAT 32
    img = np.full((size, size), fill_value=255, dtype=np.uint8)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    # CHANGED CIRCLE RADIUS TO BE MINIMUM OF 1 INSTAD OF MIN 10
    rad = np.random.randint(1, max(1, radius))
    cv2.circle(img, (col, row), rad, 0, thickness=np.random.randint(1, 5))

    # Noise
    # CHANGED NOISE BE 0 AT MIN INSTEAD OF 0.01
    # ADD NOISE RANDOMLY
    if np.random.randint(1, 4) == 1:
        img = add_salt_and_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)

    # ADD BLUR RANDOMLY
    if np.random.randint(1, 4) == 1:
        kernel_size = random.randrange(1, 8, 2)
        img = cv2.blur(img, (kernel_size, kernel_size))

    return (row, col, rad), img


def train_set():
    # TRAIN 1000 INSTEAD OF 200000
    number_of_images = 10000
    # CHANGE TO 2 INSR+TEAD OF 3.5

    with open("train_set.csv", 'w', newline='') as outFile:
        header = ['NAME', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(number_of_images):
            params, img = noisy_circle(200, 100)
            np.save("datasets/train/" + str(i) + ".npy", img)
            write(outFile, ["datasets/train/" + str(i) + ".npy", params[0], params[1], params[2]])


def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])


if __name__ == '__main__':
    train_set()
