import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from network import Net
import cv2


def find_circle(img):
    model = Net()
    # CHANGE TO THE PATH TO NEW MODEL
    checkpoint = torch.load('Circle-Detection-CNN/saved_models/50_epoch_v8_checkpoint.pth.tar')
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        image = np.expand_dims(np.asarray(img), axis=0)
        image = torch.from_numpy(np.array(image, dtype=np.float32))
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize (image)
        image = image.unsqueeze(0)
        output = model(image)

    return [round(i) for i in (200*output).tolist()[0]]


# IoU represents the success rate for how accurate the detected circle is, with 1 being complete overlap and 0 being no detection.
def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def find_circle_cnn(img, filename, dir):
    """
    Find the circle in the image, using CNN, create a new image with the detected circle highlighted in red and
    return center and radius of the circle in the form column, row, radius.

    Args:
        img (list): A 2D list representing the image.
        filename (str): A string representing the name of the image.
        dir (str): A string representing the directory to save the image to.

    Returns:
        list: the center and radius of the circle (col, row, rad), img and the img.
    """
        
    detected = find_circle(img)

    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(display_img, center=(detected[1], detected[0]), radius=detected[2], color=(0,0,255), thickness=2)
    print('CNN: ', detected)
    
    cv2.imwrite(f'{dir}/{str(filename)}_cnn_detected.jpg', display_img)

    return detected[1], detected[0], detected[2]
