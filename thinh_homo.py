# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:15:29 2019

@author: Le Quoc Thinh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Ho(img1,img2):
    im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.ORB_create()
    k1, d1 = sift.detectAndCompute(im1Gray, None)
    k2, d2 = sift.detectAndCompute(im2Gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
    matches = bf.match(d1, d2)
    matches = sorted(matches, key = lambda x: x.distance)[:int(len(matches)*0.1)]
    img3 = cv2.drawMatches(img1,k1,img2,k2,matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.axis('off')
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = k1[match.queryIdx].pt
        points2[i, :] = k2[match.trainIdx].pt
    h, mask = cv2.findHomography(points1, points2)
    height, width, channels = img2.shape
    img_new = cv2.warpPerspective(img1, h, (width, height))
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(img_new)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(img2)
    plt.axis('off')
    return img_new
 

    
if __name__ == '__main__':
    img1 = cv2.imread('1.jpg')
    img2 = cv2.imread('2.jpg')
    img_new = Ho(img2,img1)

