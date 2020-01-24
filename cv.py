#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:56:32 2020

@author: hqt98
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import os
class Pano():
    def __init__(self, path, ratio, mindist, le=True):
        filepaths = [os.path.join(path,i) for i in os.listdir(path)]
        self.images = []
        for path in filepaths:
            self.images.append(cv2.imread(path))
        self.ratio = ratio
        self.mindist = mindist
        self.le = le
    
    def featureExtractor(self, im1, im2):
        query = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        train = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        
        #Keypoint detection
        #Local invariant descriptors (SIFT, SURF, etc)
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(query,None)
        kp2, des2 = orb.detectAndCompute(train,None)
        
        return kp1, des1, kp2, des2

    def matchFeatures(self, des1, des2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < self.ratio*n.distance and m.distance < self.mindist :
                good.append(m)
        
        good = np.asarray(good)
        return good
    
    def getNumberOfMatches(self, im1, im2):
        kp1, des1, kp2, des2 = self.featureExtractor(im1, im2)
        matches = self.matchFeatures(des1, des2)
        return len(matches)
    
    def stitch(self, query_image, train_image):
        if self.le:
            k = query_image.shape[1] - int(train_image.shape[1]*0.25)
            q = query_image.copy()
            q[:,:k] = 0
            plt.imshow(q),plt.show()
            kp1, des1, kp2, des2 = self.featureExtractor(q, train_image)
        else:
            kp1, des1, kp2, des2 = self.featureExtractor(query_image, train_image)
       
        #Feature matching
        good = self.matchFeatures(des1, des2)
        if len(good) < 4:
            return []
        
        #Sort good features
        good = sorted(good, key=lambda x: x.distance)
        if len(good) > 55:
            good = good[:55]
        print('max distance: ' + str(good[-1].distance)) 
        params = dict(matchColor=(0,255,0),
        singlePointColor=None,
        flags=2)
        img3 = cv2.drawMatches(query_image,kp1,train_image,kp2,good,None,**params)
        plt.imshow(img3),plt.show()
        
        # Homography estimation using RANSAC
        if len(good) >= 4:
            dst = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            src = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        else:
            raise AssertionError("Can’t find enough keypoints.")
        print('Transformation matrix: {}'.format( H))
        # Perspective warping
        width = train_image.shape[1] + query_image.shape[1]
        height = train_image.shape[0] + query_image.shape[0]
        
        result = cv2.warpPerspective(train_image, H, (width, height))
        result[0:query_image.shape[0], 0:query_image.shape[1]] = query_image
        plt.imshow(result),plt.show()
        return result
        
    def crop(self, pano):
        # Crop black part
        # Add border
        result = cv2.copyMakeBorder(pano, 10, 10, 10, 10,
                    cv2.BORDER_CONSTANT, (0, 0, 0))
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        
        # Find contour
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Find min rect
        minRect = mask.copy()
        sub = mask.copy()
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)
            
        contours = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        			cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)
        
        # Crop according to min rect
        (x, y, w, h) = cv2.boundingRect(c)
        result = result[y:y + h, x:x + w]
        plt.imshow(result),plt.show()
        return result
        
    def createPanorama(self):
        # find leftmost image
        numberOfMatches = [0]*len(self.images)
        for pos, i in enumerate(self.images):
            # only look at left haft
            left = i[:, :i.shape[1]//2]
            for j in self.images:
                right = j[:,j.shape[1]//2:]
                
                numberOfMatches[pos] += self.getNumberOfMatches(left, right)
        leftmost = self.images.pop(np.argmin(numberOfMatches))
        
        result = leftmost
        
        while len(self.images) != 0:
            numberOfMatches = [0]*len(self.images)
            for pos, i in enumerate(self.images):
                numberOfMatches[pos] += self.getNumberOfMatches(result, i)    
            nextImage = self.images.pop(np.argmax(numberOfMatches))
            temp = self.stitch(result, nextImage)
            if temp != []:
                result = temp
            result = self.crop(result)
        plt.imshow(result),plt.show()
        return result
        
p = Pano('test/', 0.75, 30, True)
pano = p.createPanorama()
cv2.imwrite('pano.png', m)