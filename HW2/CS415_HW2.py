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

    """ 
    # Implement the Convolution filter manually
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
    """

    dst = ndimage.filters.convolve(img, kernel)

    return dst


def ImageGradient(img):
    # Build the Sobel filter of x axis and y axis
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # Apply the sobel filter to img
    img_x = ndimage.filters.convolve(img, sobel_x)
    img_y = ndimage.filters.convolve(img, sobel_y)

    # The image when the x axis sobel operator is applied
    plt.figure()
    plt.imshow(img_x, cmap = 'gray')
    plt.title('X direction gradient')
    plt.show()
    plt.savefig('X_direction.png')

    # The image when the y axis sobel operator is applied
    plt.figure()
    plt.imshow(img_y, cmap = 'gray')
    plt.title('Y direction gradient')
    plt.show()
    plt.savefig('Y_direction.png')

    """
    Personal understanding:
    the x axis sobel operator will cause a result looks "vertical"
    the y axis sobel operator will cause a result looks "horizontal"
    """

    # Apply the formula, Consider gradient of x direction and y direction as wwo right-angled edges
    # Mag = np.hypot(img_x, img_y)
    Mag = np.sqrt(np.square(img_x) + np.square(img_y))
    Theta = np.arctan2(img_x, img_y)

    return Mag, Theta


# Thinner the edges by find the local maxima
def NonmaximaSuppress(img, direct):
    # Get the length and width of the image and declare a blank img as output
    img_len, img_wid = img.shape

    # TODO      Not sure why the result if different between the following two expressions
    # dst = np.zeros((img_len, img_wid), dtype = np.int32)
    dst = np.copy(img)

    # Convert the angle's value in polar coordinates to the real values
    # Shrink the range to 0-180, b/c a value and the value plus 180 is on the same line
    # However, AS FOR MY UNDERSTANDING the direction is not the same
    angle = (direct * 180.0) / np.pi
    angle[angle < 0] += 180.0

    # Go through the img
    for i in range(1, img_len - 1):
        for j in range(1, img_wid - 1):
            # Initialize the values which should be at two sides of ridges
            left = 255
            right = 255

            # Horizontal line
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
                left = img[i - 1, j]
                right = img[i + 1, j]

            # Vertical line
            elif 67.5 <= angle[i, j] < 112.5:
                left = img[i, j - 1]
                right = img[i, j + 1]

            # Line of angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                left = img[i - 1, j - 1]
                right = img[i + 1, j + 1]

            # Line of angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                left = img[i - 1, j + 1]
                right = img[i + 1, j - 1]

            # Determine if the current processing point is a local maxima
            if img[i, j] > left and img[i, j] > right:
                dst[i, j] = img[i, j]
            else:
                dst[i, j] = 0

    return dst


def EdgeLinking(img, strong_threshold, weak_threshold):
    # Apply the given ratio to the get the threshold with a real value
    strong_threshold = img.max() * strong_threshold
    weak_threshold = img.max() * weak_threshold

    img_len, img_wid = img.shape
    dst = np.zeros((img_len, img_wid), dtype = np.int32)

    # Define the value of strong edge, weak edge, and zero edge
    strong_edge = np.int32(255)
    weak_edge = np.int32(100)
    zero_edge = np.int32(0)

    # Determine the coordinate position of the three kind of edges
    strong_i, strong_j = np.where(img >= strong_threshold)
    weak_i, weak_j = np.where((img < strong_threshold) & (img >= weak_threshold))
    zeros_i, zeros_j = np.where(img < weak_threshold)

    # Make the change to the output
    dst[strong_i, strong_j] = strong_edge
    dst[weak_i, weak_j] = weak_edge
    dst[zeros_i, zeros_j] = zero_edge

    # Go through the output we have here, and determine if, for the weak edges we are currently processing, there exists
    # any strong edges. If so, change the weak edge to strong edge. If not, change the weak edge to zero edge
    for i in range(1, img_len - 1):
        for j in range(1, img_wid - 1):
            if dst[i, j] == weak_edge:
                # 8 positions abound the edge we are currently processing
                if ((dst[i + 1, j - 1] == strong_edge) or (dst[i + 1, j] == strong_edge) or
                        (dst[i + 1, j + 1] == strong_edge) or (dst[i, j - 1] == strong_edge) or
                        (dst[i, j + 1] == strong_edge) or (dst[i - 1, j - 1] == strong_edge) or
                        (dst[i - 1, j] == strong_edge) or (dst[i - 1, j + 1] == strong_edge)):
                    dst[i, j] = strong_edge
                else:
                    dst[i, j] = zero_edge

    return dst


def main():
    # Input the images
    imgPath = "lena_gray.png"
    img = plt.imread(imgPath)

    imgPath_2 = "test.png"
    img_2 = plt.imread(imgPath_2)

    # The input is flexible
    kernel_size = int(input('Enter the kernel edges\' length: '))
    sigma = int(input('Enter value of sigma: '))

    # kernel_size = 3
    # sigma = 3

    # Apply the Gaussian Blur to the image
    GaussianSmoothed = GaussianSmooth(img, kernel_size, sigma)

    plt.figure()
    plt.imshow(GaussianSmoothed, cmap = 'gray')
    plt.title('GaussianSmoothed Image')
    plt.show()
    plt.savefig('GaussianSmoothed_image.png')

    # Compute the Gradient Magnitude and direction
    Mag, Theta = ImageGradient(GaussianSmoothed)

    plt.figure()
    plt.imshow(Mag, cmap = 'gray')
    plt.title('Gradient Magnitude')
    plt.show()
    plt.savefig('Mag.png')

    plt.figure()
    plt.imshow(Theta, cmap = 'gray')
    plt.title('Gradient Direction')
    plt.show()
    plt.savefig('Theta.png')

    # Apply NMS to edge detected image
    Mag = NonmaximaSuppress(Mag, Theta)

    plt.figure()
    plt.imshow(Mag, cmap = 'gray')
    plt.title('Gradient Magnitude After NMS')
    plt.show()
    plt.savefig('Mag_after_NMS,png')

    # Separate the points in the image into 3 groups: strong edge, weak edge, and zero edge
    # link every strong edges
    # Set the strong edge value 3 times greater than weak edge
    Linked_edges = EdgeLinking(Mag, 0.03, 0.01)

    plt.figure()
    plt.imshow(Linked_edges, cmap = 'gray')
    plt.title('Linked Edges')
    plt.show()
    plt.savefig('Linked_edges.png')


if __name__ == '__main__':
    main()
