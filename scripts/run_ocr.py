import easyocr
import os
import cv2

def run_ocr_on_images(image_dir):
    reader = easyocr.Reader(['en', 'hi'], gpu=False)
    result = {}
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(image_dir, filename))
            ocr_result = reader.readtext(img)
            texts = [item[1] for item in ocr_result]
            result[filename] = texts
    return result
