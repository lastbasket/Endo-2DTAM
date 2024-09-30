import cv2

im_path = '/home/lastbasket/code/slam/EndoScopic/gslam2d/data/EndoSLAM/Colon/Pixelwise Depths/aov_image_0000.png'
img = cv2.imread(im_path, -1)
print(img.max())
print(img.shape)
cv2.imshow('', img[..., 0])
cv2.waitKey()