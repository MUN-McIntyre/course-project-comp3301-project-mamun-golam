import cv2
import numpy as np

def find_circle_hough(img, i):

    # Apply Gaussian blur to reduce noise and help Canny edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Hough Circle Transform
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

    # Draw circles on the original image
    if circles is not None:
        if len(circles) == 1:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Draw the detected circle
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.circle(display_img, center=(circle[0], circle[1]), radius=circle[2], color=(0, 0, 255), thickness=2)
                cv2.imwrite(f'test/{str(i)}_hough_detected.jpg', display_img)

                # column, row, radius or circle
                print('trt',[circle[0], circle[1], circle[2]])
                return circle[0], circle[1], circle[2]
        print('Multiple circles detected')
        return None
    else:
        print('Hough Circle not detected')
        return None