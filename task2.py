import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def ssd(im1, im2):
    return np.sum((im1 - im2) ** 2)

def sad(im1, im2):
    return np.sum(np.abs(im1 - im2))

def normalize(im):
    im0 = im - np.mean(im)
    return (im0)/np.linalg.norm(im0)

def ncc(im1, im2):
    return np.sum(normalize(im1) * normalize(im2))

def matchTemplate(image, template, method):
    h_temp, w_temp = template.shape
    h_im, w_im = image.shape
    res = np.zeros(shape=(h_im - h_temp + 1, w_im - w_temp + 1))
    for x in range(res.shape[1]):
        for y in range(res.shape[0]):
            img_part = image[y:y + h_temp, x:x + w_temp]
            res[y, x] = method(img_part, template)
    return res

img = cv.imread('0001.jpg', 0)
img2 = img.copy()
template = cv.imread('template.png', 0)
h, w = template.shape

list_of_methods = ['ssd', 'ncc', 'sad']
for meth in list_of_methods:
    img = img2.copy()
    method = eval(meth)
    matched = matchTemplate(img, template, method)
    if meth == 'ncc':
        ind = np.unravel_index(np.argmax(matched, axis=None), matched.shape)
    else:
        ind = np.unravel_index(np.argmin(matched, axis=None), matched.shape)
    top_left = ind[::-1]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(matched,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def ssd(im1, im2):
    return np.sum((im1 - im2) ** 2)

def sad(im1, im2):
    return np.sum(np.abs(im1 - im2))

def normalize(im):
    im0 = im - np.mean(im)
    return (im0)/np.linalg.norm(im0)

def ncc(im1, im2):
    return np.sum(normalize(im1) * normalize(im2))

def matchTemplate(image, template, method):
    h_temp, w_temp = template.shape
    h_im, w_im = image.shape
    res = np.zeros(shape=(h_im - h_temp + 1, w_im - w_temp + 1))
    for x in range(res.shape[1]):
        for y in range(res.shape[0]):
            img_part = image[y:y + h_temp, x:x + w_temp]
            res[y, x] = method(img_part, template)
    return res

img = cv.imread('0001.jpg', 0)
img2 = img.copy()
template = cv.imread('template.png', 0)
h, w = template.shape

list_of_methods = ['ssd', 'ncc', 'sad']
for meth in list_of_methods:
    img = img2.copy()
    method = eval(meth)
    matched = matchTemplate(img, template, method)
    if meth == 'ncc':
        ind = np.unravel_index(np.argmax(matched, axis=None), matched.shape)
    else:
        ind = np.unravel_index(np.argmin(matched, axis=None), matched.shape)
    top_left = ind[::-1]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(matched,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
