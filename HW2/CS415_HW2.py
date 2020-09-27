"""
Xichen Liu
CS415
Canny edge detector
"""

from scipy import ndimage
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

    # Implement the Convolution filter manually
    # # Get the length and width of image
    # i_rows = len(img)
    # i_cols = len(img[0])
    #
    # # flip the kernel
    # for i in range(kernel_size // 2):
    #     kernel[i], kernel[kernel_size - i - 1] = kernel[kernel_size - 1 - i], kernel[i]
    # for j in range(kernel_size):
    #     for k in range(kernel_size // 2):
    #         kernel[j][k], kernel[j][kernel_size - k - 1] = kernel[j][kernel_size - k - 1], kernel[j][k]
    #
    # # implement the pad of the boarder of image
    # pad = (kernel_size - 1) // 2
    # img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    #
    # # Initialized output image
    # dst = np.zeros((i_rows, i_cols), dtype = "float32")
    # # get the pad rid of the desired output
    # for i in np.arange(pad, i_rows + pad):
    #     for j in np.arange(pad, i_cols + pad):
    #         # The area from original image which is applied by the kernel
    #         sample_from_img = img[i - pad: i + pad + 1, j - pad: j + pad + 1]
    #         k = (sample_from_img * kernel).sum()
    #         dst[i - pad, j - pad] = k

    dst = ndimage.filters.convolve(img, kernel)

    return dst


def ImageGradient(img):
    # Build the Sobel filter of x axis and y axis
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # Apply the sobel filter to img
    img_x = ndimage.filters.convolve(img, sobel_x)
    img_y = ndimage.filters.convolve(img, sobel_y)

    # # The image if the x axis sobel operator is applied
    # plt.figure()
    # plt.imshow(img_x, cmap = 'gray')
    # plt.title('X direction gradient')
    # plt.show()
    #
    # # The image if the y axis sobel operator is applied
    # plt.figure()
    # plt.imshow(img_y, cmap = 'gray')
    # plt.title('Y direction gradient')
    # plt.show()

    """
    Personal understanding:
    the x axis sobel operator will cause a result looks "vertical"
    the y axis sobel operator will cause a result looks "horizontal"
    """

    # Apply the formula, Consider gradient of x direction and y direction as wwo right-angled edges
    Mag = np.hypot(img_x, img_y)
    Theta = np.arctan2(img_x, img_y)

    return Mag, Theta


# Thinner the edges by find the local maxima
def NonmaximaSuppress(img, dir):
    img_len, img_wid = img.shape
    dst = np.zeros(img_len, img_wid)


# This function should be deleted     TODO
def test():
    print()
    print("-------------------------------")
    print("The following is just for test")

    s = np.array([1, 1, 2, 3, 4, 5, 7])
    s[s > 2] += 20
    print(s)

    print("The above is just for test")
    print("-------------------------------")
    print()


def main():
    # Should be deleted     TODO
    test()  # Should be deleted     TODO
    # Input the images
    imgPath = "lena_gray.png"
    img = plt.imread(imgPath)

    imgPath_2 = "test.png"
    img_2 = plt.imread(imgPath_2)

    # The input is flexible
    # kernel_size = int(input('Enter the kernel edges\' length: '))  TODO
    # sigma = int(input('Enter value of sigma: '))  TODO

    kernel_size = 3  # Should be deleted     TODO
    sigma = 3  # Should be deleted     TODO

    GaussianSmoothed = GaussianSmooth(img_2, kernel_size, sigma)
    Mag, Theta = ImageGradient(GaussianSmoothed)
    print(type(Theta))
    # plt.figure()
    # plt.imshow(Mag, cmap = 'gray')
    # plt.title('Gradient Magnitude')
    # plt.show()
    # plt.figure()
    # plt.imshow(Theta, cmap = 'gray')
    # plt.title('Gradient Direction')
    # plt.show()


if __name__ == '__main__':
    main()
