"""
Xichen Liu
CS415 Hw1
Image filtering
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

dir_path = os.path.dirname((os.path.realpath(__file__)))
sys.path.insert(0, dir_path)


# firstly, flip the kernel and apply the kernel to the neighborhoods and move along the image
def ConvolutionFilter(img, kernel, flag):
    i_rows = len(img)
    i_cols = len(img[0])
    k_rows, k_cols = np.shape(kernel)
    # flip the kernel
    for i in range(k_rows // 2):
        kernel[i], kernel[k_rows - i - 1] = kernel[k_rows - 1 - i], kernel[i]
    for j in range(k_rows):
        for k in range(k_cols // 2):
            kernel[j][k], kernel[j][k_cols - k - 1] = kernel[j][k_cols - k - 1], kernel[j][k]

    # implement the pad
    h_pad = (k_rows - 1) // 2
    w_pad = (k_cols - 1) // 2
    img = cv2.copyMakeBorder(img, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_REPLICATE)
    # for the gray image
    dst = np.zeros((i_rows, i_cols), dtype = "float32")
    # for RGB image
    dst_b = np.zeros((i_rows, i_cols), dtype = "float32")
    dst_g = np.zeros((i_rows, i_cols), dtype = "float32")
    dst_r = np.zeros((i_rows, i_cols), dtype = "float32")

    # flag implements if the input image is gray or RGB
    if flag == 1:
        b, g, r = cv2.split(img)

    # get the pad rid of the desired output
    for i in np.arange(h_pad, i_rows + h_pad):
        for j in np.arange(w_pad, i_cols + w_pad):
            if flag == 1:
                # apply the kernel to the neighborhoods
                sample_from_b = b[i - h_pad: i + h_pad + 1, j - w_pad: j + w_pad + 1]
                k_b = (sample_from_b * kernel).sum()
                sample_from_g = g[i - h_pad: i + h_pad + 1, j - w_pad: j + w_pad + 1]
                k_g = (sample_from_g * kernel).sum()
                sample_from_r = r[i - h_pad: i + h_pad + 1, j - w_pad: j + w_pad + 1]
                k_r = (sample_from_r * kernel).sum()

                dst_b[i - h_pad, j - w_pad] = k_b
                dst_g[i - h_pad, j - w_pad] = k_g
                dst_r[i - h_pad, j - w_pad] = k_r
            else:
                sample_from_img = img[i - h_pad: i + h_pad + 1, j - w_pad: j + w_pad + 1]
                k = (sample_from_img * kernel).sum()
                dst[i - h_pad, j - w_pad] = k
    # merge the R, G, B back to one image
    if flag == 1:
        dst = cv2.merge([dst_b, dst_g, dst_r])

    return dst


# same to convolution but no need to flip the kernel
def CorrelationFilter(img, kernel):
    i_rows = len(img)
    i_cols = len(img[0])
    k_rows, k_cols = np.shape(kernel)

    h_pad = (k_rows - 1) // 2
    w_pad = (k_cols - 1) // 2
    img = cv2.copyMakeBorder(img, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_REPLICATE)
    dst = np.zeros((i_rows, i_cols), dtype = "float32")

    for i in np.arange(h_pad, i_rows + h_pad):
        for j in np.arange(w_pad, i_cols + w_pad):
            sample_from_img = img[i - h_pad: i + h_pad + 1, j - w_pad: j + w_pad + 1]
            k = np.dot(sample_from_img, kernel).sum()
            dst[i - h_pad, j - w_pad] = k

    return dst


# implement the formula
def GaussianKernel(size, sigma):
    kernel = np.zeros((size, size))
    c = size // 2

    if sigma <= 0:
        sigma = ((size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * sigma ** 2
    sum_val = 0
    for i in range(size):
        for j in range(size):
            x, y = i - c, j - c

            kernel[i][j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i][j]

    kernel = kernel / sum_val

    return kernel


# get the median of each neighborhoods, and apply it to the centre
def medianFilter(img, kernel_size):
    i_rows, i_cols = np.shape(img)

    pad = (kernel_size - 1) // 2
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    dst = np.zeros((i_rows, i_cols), dtype = "float32")

    img_array = []
    for i in range(pad, i_rows - pad):
        for j in range(pad, i_cols - pad):
            for p in range(kernel_size):
                for q in range(kernel_size):
                    img_array.append(img[i + p, j + q])

            img_array.sort()
            dst[i, j] = img_array[(len(img_array) + 1) // 2]
            del img_array[:]

    return dst


def main():
    imgPath = "lena.png"
    img = plt.imread(imgPath)
    print(type(img))

    imgPath_2 = "art.png"
    img_2 = plt.imread(imgPath_2)
    print(type(img_2))

    # Q1.1. apply mean filter to lena.png
    print('Mean filter(convolution) on lena.png:')
    meanKernelLen = int(input('Enter the size fo mean filter kernel\'s length(odd): '))
    meanKernelWid = int(input('Enter the size fo mean filter kernel\'s width(odd): '))
    meanKernel = np.ones((meanKernelLen, meanKernelWid), dtype = np.float32) / (meanKernelLen * meanKernelWid)
    # meanKernel = np.ones((9, 9), dtype = np.float32) / 81
    meanConvolution = ConvolutionFilter(img, meanKernel, 1)
    # plt.imshow(meanConvolution)
    # plt.show()

    # Q1.2. apply Gaussian filter to lena.png
    print('Gaussian filter(convolution) on lena.png:')
    GaussianKernelSize = int(input('Enter the size fo Gaussian Kernel edge\'s length(odd): '))
    GaussianKernelSigma = int(input('Enter the sigma fo Gaussian Kernel: '))
    GKernel = GaussianKernel(GaussianKernelSize, GaussianKernelSigma)
    # GKernel = GaussianKernel(5, 5)
    GaussianConvolution = ConvolutionFilter(img, GKernel, 1)
    # plt.imshow(GaussianConvolution)
    # plt.show()

    # Q1.3. apply sharpen filter to lena.png
    print('Sharpen filter(convolution) on lena.png:')
    sharpenKernelLen = int(input('Enter the size fo sharpen filter kernel\'s length(odd): '))
    sharpenKernelWid = int(input('Enter the size fo sharpen filter kernel\'s width(odd): '))
    # fix the sharpen degree to 5
    sharpenKernel = np.zeros((sharpenKernelLen, sharpenKernelWid), dtype = np.float32)
    sharpenKernelLenC = sharpenKernelLen // 2
    sharpenKernelWidC = sharpenKernelWid // 2
    sharpenKernel[sharpenKernelLenC, sharpenKernelWidC] = 5

    for i in range(sharpenKernelLen):
        for j in range(sharpenKernelWid):
            if sharpenKernel[i][j] != 5:
                sharpenKernel[i][j] = -4 / (sharpenKernelLen * sharpenKernelWid)

    sharpenConvolutionFilter = ConvolutionFilter(img, sharpenKernel, 1)
    # plt.imshow(sharpenConvolutionFilter)
    # plt.show()

    # Q2.1 apply mean filter to art.png by convolution filter
    print('Mean filter(convolution) on art.png:')
    meanKernelLen_2 = int(input('Enter the size fo mean filter kernel\'s length(odd): '))
    meanKernelWid_2 = int(input('Enter the size fo mean filter kernel\'s width(odd): '))
    meanKernel_2 = np.ones((meanKernelLen_2, meanKernelWid_2), dtype = np.float32) / (meanKernelLen_2 * meanKernelWid_2)
    # meanKernel_2 = np.ones((9, 9), dtype = np.float32) / 81
    meanConvolution_2 = ConvolutionFilter(img_2, meanKernel_2, 0)
    # plt.imshow(meanConvolution_2)
    # plt.show()

    # Q2.2 apply mean filter to art.png by cross-correlation filter
    print('Mean filter(correlation) on art.png:')
    meanKernelLen_2_corr = int(input('Enter the size fo mean filter kernel\'s length(odd): '))
    meanKernelWid_2_corr = int(input('Enter the size fo mean filter kernel\'s width(odd): '))
    meanKernel_2_corr = np.ones((meanKernelLen_2_corr, meanKernelWid_2_corr),
                                dtype = np.float32) / (meanKernelLen_2_corr * meanKernelWid_2_corr)
    meanCorrelation_2 = CorrelationFilter(img_2, meanKernel_2_corr)
    # plt.imshow(meanCorrelation_2)
    # plt.show()

    # Q2.3 apply median filter to art.png
    print('Median filter on art.png:')
    medianKernelLen_2 = int(input('Enter the size fo median filter kernel\'s length(odd): '))
    medianFilterPic = medianFilter(img_2, medianKernelLen_2)
    # plt.imshow(medianFilterPic)
    # plt.show()

    plt.figure()
    plt.imshow(img)
    plt.title('Origin')
    plt.show()

    plt.figure()
    plt.imshow(meanConvolution)
    plt.title('Mean Filter')
    plt.show()

    plt.figure()
    plt.imshow(GaussianConvolution)
    plt.title('Gaussian Filter')
    plt.show()

    plt.figure()
    plt.imshow(sharpenConvolutionFilter)
    plt.title('Sharpen Filter')
    plt.show()

    plt.figure()
    plt.imshow(img_2)
    plt.title('Origin')
    plt.show()

    plt.figure()
    plt.imshow(meanConvolution_2)
    plt.title('Mean Filter 2')
    plt.show()

    plt.figure()
    plt.imshow(meanCorrelation_2)
    plt.title('Mean Filter(correlation) 2')
    plt.show()

    plt.figure()
    plt.imshow(medianFilterPic)
    plt.title('Median Filter 2')
    plt.show()


if __name__ == '__main__':
    main()
