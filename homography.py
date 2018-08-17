#!/usr/bin/env python
 
import cv2
import numpy as np
 
import sys

#path to the 1st image
filename1 = sys.argv[1]
#path to the 2st image
filename2 = sys.argv[2]


if __name__ == '__main__' :

    ##First part
 
    # Read source image.
    im_src = cv2.imread(filename1)
    # Four corners of the book in source image
    pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])
 
 
    # Read destination image.
    im_dst = cv2.imread(filename2)
    # Four corners of the book in destination image.
    pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])
 
    # Calculate first Homography
    h1, status = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h1, (im_dst.shape[1],im_dst.shape[0]))

    ##Second part

    img1 = im_out
    img2 = im_dst

    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # match features using brute force matcher
    bf = cv2.BFMatcher()
    #matches = bf.knnMatch(des1,des2, k=2)
    matches = bf.match(des1,des2)

    matches = sorted(matches, key = lambda x:x.distance)

    # store all the good matches as per Lowe's ratio test.
    #good = []
    #for m,n in matches:
    #    if m.distance < 0.7*n.distance:
    #        good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    # Calculate second Homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    # Warp source image to destination based on homography
    img3 = cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))

    # Draw the five best matches
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:5],None)
     
    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
    cv2.imshow("Final Warped Image", img3)

    # Construct general homography matrix

    cv2.waitKey(0)
