import numpy as np
import cv2 as cv
import glob
import os
from matplotlib import pyplot as plt

def warp(image, p):
    p_matrix = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]], dtype=np.float32)
    return cv.warpAffine(image, p_matrix, image.shape[::-1])

def get_roi(image, region_of_interest):
    return image[region_of_interest[1]:region_of_interest[3], region_of_interest[0]:region_of_interest[2]]

def jacobian(x,y):
    return np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]], dtype=np.float32)

# imageFolder = "/home/tetiana/Documents/UCU_studies/comp_viz/Session2/Coke/img/"
# imageList = sorted(glob.glob(os.path.join(imageFolder, '*.jpg')))
# template = cv.imread(imageList.pop(0), 0)
template = cv.imread('0001.jpg', 0)

# center of the roi and half-sizes of the roi
center = [320, 195]
half_window = [50, 50]
# top-left and bottom-right corners of the roi
roi = [center[0] - half_window[0], center[1] - half_window[1], center[0] + half_window[0], center[1] + half_window[1]]

img = cv.imread('0002.jpg', 0)
temp_roi = get_roi(template, roi)

p = np.array([0,0,0,0,0,0], dtype=np.float32)
dp = np.array([1,1,1,1,1,1], dtype=np.float32)

while np.linalg.norm(dp) > 0.1:
    error = temp_roi - get_roi(warp(img, p), roi)

    imx = cv.Sobel(img, ddepth=-1, dx=1, dy=0, ksize=1)
    imy = cv.Sobel(img, ddepth=-1, dx=0, dy=1, ksize=1)
    print(imy)
    warped_imx = get_roi(warp(imx, p), roi)
    warped_imy = get_roi(warp(imy, p), roi)

    steepest_images = [np.zeros(shape=temp_roi.shape) for _ in range(6)]

    for y in range(temp_roi.shape[0]):
        for x in range(temp_roi.shape[1]):
            curr_values = np.array([warped_imx[y,x], warped_imy[y,x]]).dot(jacobian(x+roi[0], y+roi[1]))
            for i in range(6):
                steepest_images[i][y,x] = curr_values[i]

    plt.imshow(steepest_images[3], cmap='gray')
    plt.show()
    steep_im_transp = np.array([steepest_images[i].flatten() for i in range(6)])
    hessian = steep_im_transp.dot(steep_im_transp.transpose())
    sd_params = steep_im_transp.dot(error.flatten())

    dp = np.linalg.inv(hessian).dot(sd_params)
    p += dp


    # top_left = tuple(roi[:2])
    # bottom_right = tuple(roi[2:])
    # cv.rectangle(img, top_left, bottom_right, 255, 2)
    # plt.imshow(img, cmap='gray')
    # plt.show()


