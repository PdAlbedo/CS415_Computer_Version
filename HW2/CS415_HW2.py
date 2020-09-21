"""
Xichen Liu
CS415
Canny edge detector on gray pics
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

dir_path = os.path.dirname((os.path.realpath(__file__)))
sys.path.insert(0, dir_path)


# Apply Gaussian filter to the image first
def GaussianSmooth(img, kernel_size, sigma):
    # Here generate a Gaussian Kernel first.
    kernel = np.zeros((kernel_size, kernel_size))
    c = kernel_size // 2

    if sigma <= 0:
        # Set sigma default to 1
        sigma = 1

    s = 2 * sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - c, j - c

            kernel[i][j] = np.exp(-(x ** 2 + y ** 2) / s) * (1 / (2.0 * np.pi * sigma ** 2))

            sum_val += kernel[i][j]

    kernel = kernel / sum_val

    # Get the length and width of image
    i_rows = len(img)
    i_cols = len(img[0])

    # flip the kernel
    for i in range(kernel_size // 2):
        kernel[i], kernel[kernel_size - i - 1] = kernel[kernel_size - 1 - i], kernel[i]
    for j in range(kernel_size):
        for k in range(kernel_size // 2):
            kernel[j][k], kernel[j][kernel_size - k - 1] = kernel[j][kernel_size - k - 1], kernel[j][k]

    # implement the pad of the boarder of image
    pad = (kernel_size - 1) // 2
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # Initialized output image
    dst = np.zeros((i_rows, i_cols), dtype = "float32")
    # get the pad rid of the desired output
    for i in np.arange(pad, i_rows + pad):
        for j in np.arange(pad, i_cols + pad):
            # The area from original image which is applied by the kernel
            sample_from_img = img[i - pad: i + pad + 1, j - pad: j + pad + 1]
            k = (sample_from_img * kernel).sum()
            dst[i - pad, j - pad] = k

    return dst


def main():
    # Input the images
    imgPath = "lena_gray.png"
    img = plt.imread(imgPath)

    imgPath_2 = "test.png"
    img_2 = plt.imread(imgPath_2)

    kernel_size = int(input('Enter the kernel edges\' length: '))
    sigma = int(input('Enter value of sigma: '))
    GaussianSmoothed = GaussianSmooth(img, kernel_size, sigma)
    # plt.imshow(GaussianSmoothed, cmap = 'gray')
    # plt.show()


if __name__ == '__main__':
    main()
