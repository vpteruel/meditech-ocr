import pytesseract
from PIL import Image
import cv2
import numpy as np

# Load the image
image = cv2.imread("tesstrain/data/Meditech-ground-truth/eng_000000.tif")


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    # return cv2.medianBlur(image, 5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


# Preprocess the image
gray_image = get_grayscale(image)
# denoised_image = remove_noise(gray_image)
# binary_image = thresholding(denoised_image)

# thresh_image = thresholding(gray_image)
# dilated_image = dilate(gray_image)
# eroded_image = erode(gray_image)
# opening_image = opening(gray_image)
canny_image = canny(gray_image)

# save the processed images
# cv2.imwrite("dataset/gray_image.png", gray_image)
# cv2.imwrite("dataset/denoised_image.png", denoised_image)
# cv2.imwrite("dataset/binary_image.png", binary_image)
# cv2.imwrite("dataset/thresh_image.png", thresh_image)
# cv2.imwrite("dataset/dilated_image.png", dilated_image)
# cv2.imwrite("dataset/eroded_image.png", eroded_image)
# cv2.imwrite("dataset/opening_image.png", opening_image)
# cv2.imwrite("dataset/canny_image.png", canny_image)

# Perform OCR
custom_config = r'--oem 3 --psm 6'
# text1 = pytesseract.image_to_string(gray_image, config=custom_config)
# text2 = pytesseract.image_to_string(denoised_image, config=custom_config)
# text3 = pytesseract.image_to_string(binary_image, config=custom_config)

# text4 = pytesseract.image_to_string(thresh_image, config=custom_config)
# text5 = pytesseract.image_to_string(dilated_image, config=custom_config)
# text6 = pytesseract.image_to_string(eroded_image, config=custom_config)
# text7 = pytesseract.image_to_string(opening_image, config=custom_config)
text8 = pytesseract.image_to_string(canny_image, config=custom_config)

print("Text........: 01JR8667B9XFSC9FGF063TR6S2")
# print("Gray........:", text1.strip())
# print("Denoised....:", text2.strip())
# print("Binary......:", text3.strip())
# print("Threshold...:", text4.strip())
# print("Dilated.....:", text5.strip())
# print("Eroded......:", text6.strip())
# print("Opening.....:", text7.strip())
print("Canny.......:", text8.strip())

# identify and draw bounding boxes around each letter in detected text
# for detection in text8:
#     x, y, w, h = cv2.boundingRect(detection)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Detected Text", image)
