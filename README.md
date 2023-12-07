# CNN vs Hough Circle Detection Comparison

This repository compares the Deep Convolutional Neural Network and Hough Circles when it comes to Circle detection. This repository contains statistics on a test conducted on 1000 images when varying levels of noise and blur. CNN and Hough were both used to detect the circles in these 1000 images and the resulting pictures can be found in the folder called *test*. The statistics of these tests, containing the average success rate of each algorithm and the average execution time is contained in the file named *stats.txt*.


## Statistics on 1000 images

```
Average IoU (CNN): 0.8616750749654489
Average IoU (Hough): 0.7195148291577159
Average Execution Time (CNN): 0.0707612841129303 seconds
Average Execution Time (Hough): 0.007999287843704224 seconds
```


## Installation

Clone the repository:

```bash
git clone https://github.com/MUN-McIntyre/course-project-comp3301-project-mamun-golam.git
cd circle-detection-cnn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```


## compare_detection_stats.py

This code generates 1000 random test images, with varying levels of noise and blur, and run both CNN and Hough Circle Detection on the test images. It calculates the success rate of each algorithm using Intersection over Union for "success" rate and the execution time. And finally it prints out the average IoU and execution time for both algorithms.

- To run this code, run the following in terminal (make sure there is a folder named 'test' main directory):

```bash
python compare_detection.py
```

- To change the number of images used in comparison, change the get_stats(no_of_images) parameter in the main() function of the code.


## detecting_real_circles.py

This code used to run and visualize the comparison between both algorithms on real life circle examples. It loops thourgh a folder containing real circle images ('real_test' in this case). It runs both CNN and Hough Circle Detection on the images in this folder, and saves the new image with the red detected circle on top of the original image in the same folder with a name with the format *imagename_algorithm_detected.jpg*.

- To run this code, run the following in terminal:

```bash
python detecting_real_circles.py
```

- To change the folder where your real images are saved, change the parameter to find_real_circle('real_test') to a path to the folder you want.


## detecting_real_circles.py

This code is used generate a circle on a 200x200 grayscale image using CV2 and add random noise and blurriness to the image. It takes a filename for the output image and saves the image in the *test* folder. It returns the center and radius of the circle and the img list itself, in the format: (col, row, rad), img.

- The draw_circle(filename) function in this code is used to genereate images for testing in the compare_detection_stats.py code.


## test

This folder contains the images that were used to compare statistics for both algorithms.


## real_test

This folder contains a few examples of the circle examples in real life and contains how both algorithms worked on these images.