import cv2
import numpy as np

def find_circle_hough(img, filename, dir):
    """
    Find the circle in the image, using Hough Circles and Canny edge, create a new image with the detected 
    circle highlighted in red and return center and radius of the circle in the form column, row, radius.

    Args:
        img (list): A 2D list representing the image.
        filename (str): A string representing the name of the image.
        dir (str): A string representing the directory to save the image to.

    Returns:
        list: the center and radius of the circle (col, row, rad), img and the img. Returns None if the
        circles detected are more than 0 or more than 1.
    """

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Find the circles using Hough Circles Transform
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0
    )

    if circles is not None:
        if len(circles) == 1:

            # Round all the foudnd values to nearest integer
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Draw the detected circle in red color
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.circle(display_img, center=(circle[0], circle[1]), radius=circle[2], color=(0, 0, 255), thickness=2)
                cv2.imwrite(f'{dir}/{str(filename)}_hough_detected.jpg', display_img)
                print('Hough: ',[circle[0], circle[1], circle[2]])

                return circle[0], circle[1], circle[2]
            
        print('Multiple circles detected')
        return None
    else:
        print('Hough Circle not detected')
        return None