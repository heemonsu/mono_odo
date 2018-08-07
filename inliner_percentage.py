#!/usr/bin/env python
 
import cv2
import numpy as np
import sys

#path to the 1st image
filename1 = sys.argv[1]
#path to the 2st image
filename2 = sys.argv[2]

def intersection(lst1, lst2):
    lst3 = [list(filter(lambda x: x in lst1, sublist)) for sublist in lst2]
    #lst3 = [filter(lambda x: x in lst1, sublist) for sublist in lst2]
    return lst3

if __name__ == '__main__' :

    ##First part
 
    # Read source images.
    im_src = cv2.imread(filename1)
    im_dst = cv2.imread(filename2)


    # feature extractors and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()
    fast = cv2.FastFeatureDetector()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = sift.detectAndCompute(im_src,None)
    kp2, des2 = sift.detectAndCompute(im_dst,None)

    # find the keypoints with STAR
    #kp = fast.detect(im_src_0,None)

    # compute the descriptors with BRIEF
    #kp, des = brief.compute(img, kp)

    # match features using brute force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    matches_bis = bf.knnMatch(des2,des1, k=2)
    #matches = bf.match(des1,des2)
    #matches_bis = bf.match(des2,des1)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    #matches = flann.knnMatch(des1,des2,k=2)
    #matches_bis = flann.knnMatch(des2,des1,k=2)

    good = []
    for m,n in matches :
        if m.distance < 0.7*n.distance :
            good.append(m)

    good_bis = []
    for m,n in matches_bis :
        if m.distance < 0.7*n.distance :
            good_bis.append(m)


    pts_src = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    pts_dst = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    pts_src_bis = np.float32([ kp2[m.queryIdx].pt for m in good_bis ]).reshape(-1,1,2)
    pts_dst_bis = np.float32([ kp1[m.trainIdx].pt for m in good_bis ]).reshape(-1,1,2)


    H1, status1 = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    H2, status2 = cv2.findHomography(pts_src_bis, pts_dst_bis, cv2.RANSAC, 5.0)

    inliner1 = 0
    inliner2 = 0

    for x in (status1):
        if(x > 0):
            inliner1 = inliner1 +1

    for x in (status2):
        if(x > 0):
            inliner2 = inliner2 +1

    print(inliner1)
    print(len(matches))

    print(inliner2)
    print(len(matches_bis))

    percentage_1 = float(inliner1)/float(len(matches))
    percentage_2 = float(inliner2)/float(len(matches_bis))

    print("%0.4f" % percentage_1)
    print("%0.4f" % percentage_2)

    percentage = (float(percentage_1) + float(percentage_2))/2.0

    print("%0.4f" % percentage)






