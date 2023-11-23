import sys
import time
import numpy as np

sys.path.append('Hough_Circle_Detection')
sys.path.append('Circle-Detection-CNN')
from canny_circle_detection import find_circle_hough
from validation import find_circle_cnn, iou
from create_circles import draw_circle


def get_stats(no_of_images):
    """
    Generates images and runs both CNN and Hough on it to compare the accuracy rate and execution time. Prints iut the stats.

    Args:
        no_of_images (int): Number of images to generate.
    """

    cnn_results = []
    hough_results = []
    total_cnn_time = 0
    total_hough_time = 0

    for i in range(no_of_images):
        params, img = draw_circle(i)

        # Measure execution time for find_circle_hough
        start_time = time.time()
        hough_detect = find_circle_hough(img, i)
        hough_execution_time = time.time() - start_time
        total_hough_time += hough_execution_time

        # Measure execution time for find_circle_cnn
        start_time = time.time()
        cnn_detect = find_circle_cnn(img, i)
        cnn_execution_time = time.time() - start_time
        total_cnn_time += cnn_execution_time


        if cnn_detect != None:
            result_cnn = iou(params, cnn_detect)
        else:
            result_cnn = 0
        if hough_detect != None:
            result_hough = iou(params, hough_detect)
        else:
            result_hough = 0

        
        cnn_results.append(result_cnn)
        hough_results.append(result_hough)

    average_cnn_time = total_cnn_time / no_of_images
    average_hough_time = total_hough_time / no_of_images

    print("Average IoU (CNN):", np.average(cnn_results))
    print("Average IoU (Hough):", np.average(hough_results))
    print("Average Execution Time (CNN):", average_cnn_time, "seconds")
    print("Average Execution Time (Hough):", average_hough_time, "seconds")


def main():
    get_stats(1000)


if __name__ == '__main__':
    main()

