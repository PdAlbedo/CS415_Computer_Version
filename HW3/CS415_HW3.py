"""
Xichen Liu
CS415 HW3
line detection by hough transform
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname((os.path.realpath(__file__)))
sys.path.insert(0, dir_path)


def line_detection_hough_transform(image_o, img, sz, sig, rho_int, theta_int, img_path):
    img_len, img_wid = img.shape
    # Calculate the max value of rho
    rho_limit = round(np.sqrt(np.square(img_len) + np.square(img_wid)))

    # Determine the range of rho and theta in hough space
    rho_range = np.arange(-rho_limit, rho_limit, rho_int)
    theta_range = np.arange(0, 180, theta_int)

    # Calculate the cos and sin of radians
    cos_thetas_in_pi = np.cos(np.deg2rad(theta_range))
    sin_thetas_in_pi = np.sin(np.deg2rad(theta_range))

    # Initialize accumulator to all zeros
    accumulator = np.zeros((len(rho_range), len(theta_range)))

    title = 'Image: ' + str(img_path) + '; Gaussian kernel size: ' + str(sz) + '; Sigma: ' + str(sig) + \
            '; Rho axis segments: ' + str(rho_int) + '; Theta axis segments: ' + str(theta_int)
    file_name = 'Results/Image_' + str(img_path).partition('.')[0] + '_Sig_' + str(sig) + '_Int_' \
                + str(rho_int) + '.png'

    figure = plt.figure(figsize = (12, 6))
    figure.suptitle(title)
    subplot1 = figure.add_subplot(1, 4, 1)
    subplot1.imshow(image_o)
    subplot2 = figure.add_subplot(1, 4, 2)
    subplot2.imshow(img, cmap = "gray")
    subplot3 = figure.add_subplot(1, 4, 3)
    subplot3.set_facecolor((0, 0, 0))

    # For every points in the original image, apply the votes
    for x in range(img_wid):
        for y in range(img_len):
            if img[y][x] != 0:
                x_axis, y_axis = [], []
                for theta_idx in range(len(theta_range)):
                    rho = int(round(x * cos_thetas_in_pi[theta_idx]) + (y * sin_thetas_in_pi[theta_idx]))
                    theta = theta_range[theta_idx]
                    # Voting
                    accumulator[rho][theta_idx] += 1
                    x_axis.append(theta)
                    y_axis.append(rho)
                subplot3.plot(x_axis, y_axis, color = "white", alpha = 0.05)

    # Threshold some high values then draw the line
    # As for this part, after I tried some threshold values, 50 might be a good one
    edges = np.where(accumulator > 50)
    positions = list(zip(edges[0], edges[1]))

    # Use line equation to draw detected line on an original image
    for i in range(0, len(positions)):
        a = np.cos(np.deg2rad(positions[i][1]))
        b = np.sin(np.deg2rad(positions[i][1]))
        x0 = a * positions[i][0]
        y0 = b * positions[i][0]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        img = cv2.line(image_o, (x1, y1), (x2, y2), (0, 255, 0), 1)

    subplot4 = figure.add_subplot(1, 4, 4)
    subplot4.imshow(img)
    subplot1.title.set_text("Origin")
    subplot2.title.set_text("Edge")
    subplot3.title.set_text("Hough Space")
    subplot4.title.set_text("Detected Lines")
    plt.savefig(file_name)
    plt.show()


if __name__ == '__main__':

    imgPath = str(input('Enter the image you want to precess: '))
    # imgPath = 'test2.bmp'
    image_origin = cv2.imread(imgPath)

    image = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter to the image
    kernel_size = int(input('Enter a kernel border length: '))
    sigma = int(input('Enter a sigma value: '))
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Apply Canny edge detector to the image
    image = cv2.Canny(image, 10, 30)

    # Apply line detection by hough transform
    rho_interval = float(input('Enter a value for rho axis segments in hough space: '))
    theta_interval = float(input('Enter a value for theta axis segments in hough space: '))
    line_detection_hough_transform(image_origin, image, kernel_size, sigma, rho_interval, theta_interval, imgPath)
