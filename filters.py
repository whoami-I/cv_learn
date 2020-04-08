import cv2 as cv
import numpy as np

baseUrl = '/local/sdb/'
img = cv.imread(baseUrl + 'img.jpg', cv.IMREAD_UNCHANGED)

# 均值滤波
result = cv.blur(img, (60, 60))
cv.namedWindow('blur', cv.WINDOW_KEEPRATIO)
cv.imshow('blur', result)

# 方框滤波
result = cv.boxFilter(img, -1, (60, 60), normalize=1)
cv.namedWindow('boxFilter', cv.WINDOW_KEEPRATIO)
cv.imshow('boxFilter', result)

cv.waitKey(0)
cv.destroyAllWindows()
