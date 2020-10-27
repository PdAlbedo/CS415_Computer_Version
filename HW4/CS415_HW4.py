"""
Xichen Liu
CS415 HW4
Histogram-based Skin Color Detection
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

dir_path = os.path.dirname((os.path.realpath(__file__)))
sys.path.insert(0, dir_path)


# Drawing the histogram
def histo_process(img, histo):
    # Obtain the interval of the histogram
    row = histo.shape[0]
    h_block = 255 // row + 1
    col = histo.shape[1]
    s_block = 255 // col + 1

    # Convert img to HSV space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    # Obtain the length and width of the image we need to go through
    length = h.shape[0]
    width = h.shape[1]

    # Stack the Vote of the (H, S) pairs appearance up
    for i in range(length):
        for j in range(width):
            h_val = h[i][j]
            s_val = s[i][j]
            histo[h_val // h_block][s_val // s_block] += 1

    return histo


# Apply the made histogram to the test images
def apply_histo_model(img, histo, thresho):
    # Obtain the interval of the histogram
    row = histo.shape[0]
    h_block = 255 // row + 1
    col = histo.shape[1]
    s_block = 255 // col + 1

    # Convert img to HSV space and obtain the length and width of the image we need to go through
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    length = img.shape[0]
    width = img.shape[1]

    # Go through the image
    for p in range(length):
        for q in range(width):
            h_val = h[p][q]
            s_val = s[p][q]

            # Determine if the value passed the threshold
            if histo[h_val // h_block][s_val // s_block] < thresho:
                # If not pass the threshold, convert the pixel to black
                img[p][q] = [0, 0, 0]

    # Convert the image back to BGR space, so it can be shown normally
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img


def Gaussian_model(histo):
    G_mean = np.mean(histo)
    G_convar = np.cov(histo)

    pass


if __name__ == '__main__':

    # Code to customize the input
    # hs_histo_row = int(input('Enter the Hue quantization level of histogram: '))
    # hs_histo_rows = [hs_histo_row]
    # hs_histo_col = int(input('Enter the Saturation quantization level of histogram: '))
    # hs_histo_cols = [hs_histo_col]
    # threshold = int(input('Enter the threshold quantization level of filter: '))
    # thresholds = [threshold]

    # Pre-set input
    hs_histo_rows = [11, 150, 256]
    hs_histo_cols = [11, 150, 256]
    thresholds = [0.0001, 0.001, 0.01]

    for hs_histo_idx in range(len(hs_histo_rows)):
        hs_histo = np.zeros(shape = (hs_histo_rows[hs_histo_idx], hs_histo_cols[hs_histo_idx]))

        train_imgs = os.listdir('.\\train_imgs')

        for train_fl in train_imgs:
            train_img = cv2.imread('.\\train_imgs\\' + train_fl)
            hs_histo = histo_process(train_img, hs_histo)

        hs_histo = hs_histo / sum(hs_histo.flat)

        np.savetxt('.\\Results\\r_' + str(hs_histo_rows[hs_histo_idx]) + '_c_' + str(hs_histo_cols[hs_histo_idx]) +
                   '_histo.txt', hs_histo)

        with open('.\\Results\\r_' + str(hs_histo_rows[hs_histo_idx]) + '_c_' + str(hs_histo_cols[hs_histo_idx]) +
                  '_histo.csv', 'w+') as my_csv:
            csvWriter = csv.writer(my_csv, delimiter = ',')
            csvWriter.writerows(hs_histo)

        plt.figure()
        plt.hist(hs_histo)
        plt.title('r_' + str(hs_histo_rows[hs_histo_idx]) + '_c_' + str(hs_histo_cols[hs_histo_idx]) + '_histo')
        plt.savefig('.\\Results\\r_' + str(hs_histo_rows[hs_histo_idx]) + '_c_' + str(hs_histo_cols[hs_histo_idx]) +
                    '_histo.png')
        plt.show()

        for th_idx in range(len(thresholds)):
            test_imgs = os.listdir('.\\test_imgs')

            for test_fl in test_imgs:
                test_img = cv2.imread('.\\test_imgs\\' + test_fl)
                output = apply_histo_model(test_img, hs_histo, thresholds[th_idx])
                cv2.imshow('Output', output)
                file_name = '_r_' + str(hs_histo_rows[hs_histo_idx]) + '_c_' + str(hs_histo_cols[hs_histo_idx]) + \
                            '_th_' + str(thresholds[th_idx]) + '_' + test_fl
                cv2.imwrite(os.path.join('.\\Results\\', file_name), output)
                cv2.waitKey(0)
