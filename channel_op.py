import cv2 as cv
import numpy as np

baseUrl = '/local/sdb/'
img = cv.imread(baseUrl + 'img.jpg', cv.IMREAD_UNCHANGED)

# 获取RGB三通道的值
b, g, r = cv.split(img)
# cv.namedWindow('image', cv.WINDOW_KEEPRATIO)  # 窗口大小可以改变
# cv.namedWindow('BLUE', cv.WINDOW_KEEPRATIO)  # 窗口大小可以改变
# cv.namedWindow('GREEN', cv.WINDOW_KEEPRATIO)
# cv.namedWindow('RED', cv.WINDOW_KEEPRATIO)
# cv.imshow('BLUE', b)
# cv.imshow('GREEN', g)
# cv.imshow('RED', r)
# cv.imshow('image', img)

# 单独显示某个通道的颜色，需要使用merge操作，把其他两个通道全部置为0
b_channel = cv.split(img)[0]
g_channel = np.zeros((img.shape[0], img.shape[1]), img.dtype)
r_channel = np.zeros((img.shape[0], img.shape[1]), img.dtype)
blue = cv.merge([b_channel, g_channel, r_channel])
cv.namedWindow('blue', cv.WINDOW_KEEPRATIO)
cv.imshow('blue', blue)

code = cv.waitKey(0)
cv.destroyAllWindows()
