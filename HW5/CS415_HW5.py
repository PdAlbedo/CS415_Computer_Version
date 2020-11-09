"""
Xichen Liu
CS415 HW5
Image classification
"""

import os
import sys
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname((os.path.realpath(__file__)))
sys.path.insert(0, dir_path)


def sift_detector(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img_gray.shape)

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    cv2.drawKeypoints(img_gray, kp, img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img, kp, des

def find_nearest_node(node_list, centroid):
    distance = []
    for node in node_list:
        dis = np.linalg.norm(node - centroid)
        distance.append(dis)
    return node_list[distance.index(min(distance))]


if __name__ == '__main__':
    # add_pic = os.walk('.\\data\\train\\TallBuilding')
    visual_dict = {}
    for file_name in os.listdir('.\\data\\train\\TallBuilding'):
        image = cv2.imread('.\\data\\train\\TallBuilding\\' + file_name)

        image, key_point, descriptor = sift_detector(image)

        k_means = KMeans(n_clusters = 10, max_iter = 500, random_state = 0).fit(descriptor)
        # visual_words = k_means.cluster_centers_

        for centroids in k_means.cluster_centers_:
            nearest_node = str(find_nearest_node(descriptor, centroids))
            visual_dict[nearest_node] = visual_dict.get(nearest_node, 0) + 1

    print(visual_dict.values())
    # plt.hist(visual_dict.keys())
