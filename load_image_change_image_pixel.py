import cv2 as cv

baseUrl = '/local/sdb/'
img = cv.imread(baseUrl + 'img.jpg', cv.IMREAD_UNCHANGED)
cv.namedWindow('image', cv.WINDOW_KEEPRATIO)  # 窗口大小可以改变
cv.namedWindow('split', cv.WINDOW_KEEPRATIO)  # 窗口大小可以改变
# for i in range(len(img)):
#     for j in range(len(img[i])):
#         img[i, j, 1] = 30
# 修改像素的值
# img[0:len(img) - 500, 0:len(img) - 100] = [255, 255, 255]

# 获取img的形状，为一个turple
print(img.shape)

# 获取特定区域
top = int(len(img) / 4)
bottom = int(len(img) * 3 / 4)
left = int(len(img[0]) / 4)
right = int(len(img[0]) * 3 / 4)
face = img[top:bottom, left:right]
cv.imshow('split', face)
cv.imshow('image', img)
code = cv.waitKey(0)
cv.destroyAllWindows()
