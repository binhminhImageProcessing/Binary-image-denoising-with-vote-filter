# -*- coding: utf-8 -*-
"""
This is a simple implementation of vote filter for binary image
@author: Binh Minh
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import random

def vote_filter (image, t, window_size):
    '''
    input:
        image:         image with noise
        t    :         a threshold number, must be large than half of window size
        window_size:   the size of scan window 
        
    output:
        filtered image
    '''
    image_bin = image.copy()
    image_bin [image == 255] = 1
    # First we do zero padding to the image, with the image is at the middle
    new_image = np.zeros((image.shape[0] + window_size - 1, image.shape[1] + window_size - 1))
    new_image[window_size - 2:image.shape[0] + window_size - 2, window_size - 2:image.shape[1] + window_size - 2] = image_bin
    
    # Define a scan window to scan
    window = np.ones((window_size, window_size))
    window [int(window_size/2), int(window_size/2)] = 0 # Set the middle element to zero
    
    # Apply convolution
    conv_image = convolve2d (image_bin, window, 'same')
    
    filtered_image = conv_image.copy()
    
    # Scan to check the vote. If a value of a pixel in convolution image is larger than t,
    # it means that pixel should be 1 and other wise.
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            if filtered_image[i,j] >= t:
                filtered_image[i,j] = 1
            else:
                filtered_image[i,j] = 0
    
    return filtered_image*255
            
def main():
    image_size = 256
    window_size = 7
    t = int(window_size/2) + 1
    
    # Create sample image
    image = np.zeros((image_size,image_size)) 
    image [50:150, 50:150] = 255
    
    number = 70
    x = random.sample(range(1, image_size), number)    
    y = random.sample(range(1, image_size), number)
    
    # randomly put noise
    for i in range(len(x)):
        image[y[i], x[i]] = 255
        
    # Vote filter
    filtered_image = vote_filter (image, t, window_size)
         
    # Visualize
    plt.figure()
    plt.gray()
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title('Input image')
    plt.subplot(1,2,2)
    plt.imshow(filtered_image)
    plt.title('Denoise image')
    
main()
    
    
    
        