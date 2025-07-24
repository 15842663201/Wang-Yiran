import cv2

img = cv2.imread('/Users/xuhaoyang/Desktop/NeRF/insightface-master/IMG_1933.JPG')
if img is None:
    print("Failed to load image.")
else:
    print("Image loaded successfully.")
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()