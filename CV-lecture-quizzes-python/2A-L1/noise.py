import cv2
import numpy as np

dolphin = cv2.imread("images/dolphin.png")
dolphin = dolphin[:,:,0]

noise = np.random.rand(*dolphin.shape) * 50
result = dolphin + noise

cv2.imshow('Result', result.astype(np.uint8))
cv2.waitKey(0)

