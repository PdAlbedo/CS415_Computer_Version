"""
Xichen Liu
CS415 HW5
Image classification
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

dir_path = os.path.dirname((os.path.realpath(__file__)))
sys.path.insert(0, dir_path)


class Image:
    def __init__(self, img = None, name = None, idx = 0, key_p = None,
                 des = None, lab = None, hist = None, predict = None):
        self.image = img  # ndarray of an image
        self.image_name = name  # image file's name
        self.ID = idx  # image ID
        self.key_point = key_p  # key points for features
        self.descriptor = des  # descriptors for features
        self.label = lab  # class of the image
        self.histogram = hist  # distribution histogram
        self.predicted_label = predict  # FOR TEST IMAGES ONLY; predicted class by knn predictor


# knn classifier
class KnnClassifier:
    def __init__(self, k = 1):
        self.k = k

    def predict(self, train_imgs, test_imgs):

        for test_img in test_imgs:
            # Stores all distances between the processing image for test and every images in training set
            classify_dis = {}

            for train_img in train_imgs:
                dis = np.linalg.norm(test_img.histogram - train_img.histogram)  # Euclidean Distance
                classify_dis[str(train_img.ID)] = dis  # key of the dictionary is the image ID

            # Sort and find the nearest k training image
            classify_dis = {k: v for k, v in sorted(classify_dis.items(), key = lambda item: item[1])}

            k_imgs_labels = []  # Nearest k training image's class
            for i in range(self.k):
                nearest_idx = list(classify_dis.keys())[i]
                for j in train_imgs:
                    if j.ID == int(nearest_idx):
                        k_imgs_labels.append(j.label)

            # Find the frequency of each class
            label_frequency = {}
            for i in range(len(k_imgs_labels)):
                if k_imgs_labels[i] not in label_frequency.keys():
                    label_frequency[str(k_imgs_labels[i])] = 1
                else:
                    label_frequency[str(k_imgs_labels[i])] += 1

            label_frequency = {k: v for k, v in sorted(label_frequency.items(), key = lambda item: item[1])}

            # Assign the class which appear the most frequent to the testing image's predicted_label
            label_idx = list(label_frequency.keys())[-1]
            test_img.predicted_label = label_idx

        return test_imgs


# Update the kep point and descriptor attributes of every image
def update_images(imgs, path, flag):
    key_des = []  # Stores the tuple (key point, descriptor) for every images
    sift = cv2.SIFT_create()
    # Extract the images in the training/testing set
    for i in range(len(imgs)):
        img_gray = cv2.cvtColor(imgs[i].image, cv2.COLOR_BGR2GRAY)
        k_and_d = (sift.detectAndCompute(img_gray, None))
        imgs[i].key_point = k_and_d[0]
        imgs[i].descriptor = k_and_d[1]
        key_des.append(k_and_d)
        if flag:
            save_img = imgs[i].image.copy()
            save_img = cv2.drawKeypoints(img_gray, k_and_d[0], save_img,
                                         flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(path + imgs[i].label + '_' + imgs[i].image_name + '.jpg', save_img)
            # cv2.imshow('Output', save_img)
            # cv2.waitKey(0)

    return imgs, key_des


# Put all descriptors of all features of all images in the training set into one list
def coalesce_des(key_des):
    # Stores the descriptors only
    tmp = [key_des[i][1] for i in range(0, len(key_des)) if key_des[i][1] is not None]

    ttl_len = 0  # Amount of all descriptors
    for i in range(len(key_des)):
        ttl_len += len(key_des[i][0])

    des_num = len(tmp[0][0])  # Length of a single descriptor

    des = np.zeros((ttl_len, des_num))  # Initialize an array to store all descriptors

    c = 0
    for i in tmp:
        for j in i:
            des[c] = j
            c += 1

    return des


# cv2,kmeans
def k_means(des, words):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    com, lab, center = cv2.kmeans(des, words, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS)

    return com, lab, center


# Build distribution histograms for every images
def histogram_made(imgs, center, his_len):
    for img in imgs:
        # For every images initialize a histogram
        img.histogram = np.zeros(his_len, dtype = np.float)

        if img.descriptor is not None:
            for vector in img.descriptor:  # vector is every descriptor of a single feature
                distance_all = {}  # Stores all distances between the processing descriptor and every centroids

                for cen in center:
                    dis = np.linalg.norm(vector - cen)  # Euclidean Distance
                    distance_all[str(cen)] = dis

                # Sort to find the nearest centroid to the processing descriptor
                distance_all = {k: v for k, v in sorted(distance_all.items(), key = lambda item: item[1])}
                nearest_cen = np.array(list(distance_all.keys())[0])

                # Do the vote
                for i in range(len(center)):
                    if center[i] is nearest_cen:
                        img.histogram[i] += 1

            # Normalize the histogram
            if sum(img.histogram) != 0:
                img.histogram = img.histogram / sum(img.histogram)

        else:
            img.histogram = np.zeros(his_len, dtype = np.float)

    return imgs


def accuracy(test_imgs):
    ttl = 0
    correctness = 0
    for img in test_imgs:
        if img.label == img.predicted_label:
            ttl += 1
            correctness += 1
        else:
            ttl += 1

    score = float(correctness) / float(ttl)
    score = round(score, 3)

    return score


if __name__ == '__main__':
    start_time = time.time()

    train_images = []  # List stores all images for test
    test_images = []  # List stores all images for test
    label_name = []  # All possible labels for the images

    num_class_train = 0  # Number of the class's folder used
    idx_train = 0  # Assign every image an ID
    for dir_name_train in os.listdir('.\\data\\train'):
        if num_class_train == 10:
            break
        num_class_train += 1

        image_per_class_train = 0  # Number of images for each class used
        for file_name_train in os.listdir('.\\data\\train\\' + dir_name_train):
            if image_per_class_train == 3:
                break
            image_per_class_train += 1

            img_train = cv2.imread('.\\data\\train\\' + dir_name_train + '\\' + file_name_train)

            train_images.append(Image(img_train, os.path.splitext(file_name_train)[0],
                                      idx = idx_train, lab = dir_name_train))
            idx_train += 1

    num_class_test = 0
    idx_test = 0
    for dir_name_test in os.listdir('.\\data\\validation'):
        if num_class_test == 10:
            break
        num_class_test += 1

        image_per_class_test = 0
        for file_name_test in os.listdir('.\\data\\validation\\' + dir_name_test):
            if image_per_class_test == 1:
                break
            image_per_class_test += 1

            img_test = cv2.imread('.\\data\\validation\\' + dir_name_test + '\\' + file_name_test)

            test_images.append(Image(img_test, os.path.splitext(file_name_test)[0],
                                     idx = idx_test, lab = dir_name_test))
            idx_test += 1

    # Update the images' list
    try:
        os.mkdir('.\\Results\\train_imgs')
    except OSError as error:
        pass
    path_train = '.\\Results\\train_imgs\\'
    train_images, train_images_key_des = update_images(train_images, path_train, 1)
    test_images, test_images_key_des = update_images(test_images, path_train, 0)

    # Put all descriptors of all features of all images in the training set into one list
    descriptor = coalesce_des(train_images_key_des).astype('float32')

    clusters = [100, 250]  # Try different cluster numbers
    for cluster_num in clusters:
        # Obtain centroids of the clusters
        compact, label, centroids = k_means(descriptor, cluster_num)

        # Build distribution histograms for every images
        train_images = histogram_made(train_images, centroids, cluster_num)
        test_images = histogram_made(test_images, centroids, cluster_num)

        if cluster_num == 100:
            plt.figure()
            plt.hist(centroids, bins = 100)
            plt.savefig('.\\Results\\Visualize_100.png')

        # Try different k values for knn
        k = [1, 3, 5]
        for k_val in k:
            # Initialize a knn classifier
            knn_classifier = KnnClassifier(k = k_val)
            test_images = knn_classifier.predict(train_images, test_images)

            for i in test_images:
                try:
                    os.mkdir('.\\Results\\test_result')
                except OSError as error:
                    pass
                try:
                    os.mkdir('.\\Results\\test_result\\' + str(cluster_num) + '_clusters_' + str(k_val) + 'nn')
                except OSError as error:
                    pass
                folder_path = '.\\Results\\test_result\\' + str(cluster_num) + '_clusters_' + \
                              str(k_val) + 'nn\\' + i.predicted_label
                try:
                    os.mkdir(folder_path)
                except OSError as error:
                    pass

                file_path = folder_path + '\\' + i.label + '_' + i.image_name + '.jpg'
                cv2.imwrite(file_path, i.image)

            accuracy_score = accuracy(test_images)
            print('Number of clusters: %d; K value: %d' % (cluster_num, k_val))
            print('    Accuracy Score: ', end = '')
            print(accuracy_score)

    end_time = time.time()
    print()
    print('RunTime: ', end_time - start_time, 's')
