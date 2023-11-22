import numpy as np
import matplotlib.pyplot as plt

'''
A code to visualize .npy files containing images for train set
'''

# Load data from the npy file
data = np.load('datasets/train/2200.npy')

# Visualize the data
plt.imshow(data, cmap='gray')
plt.title('Visualized NPY Data')
plt.colorbar()
plt.show()
