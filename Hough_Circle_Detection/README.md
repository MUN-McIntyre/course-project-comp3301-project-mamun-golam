# Hough Circle Detection

This code detects circles in an image using Canny Edges and Hough Circles. It takes in an image returns the col, row and radius of the detected circle and also draws a red detected circle on top of the image.

# Working

This code first uses gaussian blur and canny edges to smooth out the image and define the edges. Then it uses Hough Circles on top of the smoothed image to detect the image. Since we are only working on detecting one circle and the images provided only contain one circle, stop the execution of the function and prints whether it detected multiple or no circles. If it successfully detects just one circle, we print out the circle col, row, radius and draw the red detected circle on the original image.

# Usage

To use this code you have to use the function find_circle_hough(img, filename, dir) in the canny_circle_detection.py. Run the find_circle_cnn(img, filename, dir) function with the image array as the img parameter and the filename and save directory of the output detected image.
