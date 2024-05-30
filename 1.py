import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import PyPDF2

def pdf_to_images(pdf_path):
    # Convert PDF to images using pdf2image
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path)
    return images

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding to preprocess the image
    processed_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed_image

def classify_text(image):
    # Perform OCR on the image
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['level'])
    classification_result = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text = d['text'][i]
        conf = int(d['conf'][i])
        if conf > 60:  # Confidence threshold to filter weak detections
            # Simple heuristic to classify text
            if any(c.isdigit() for c in text) or any(c.isalpha() for c in text):
                classification_result.append(("Printed", text, conf, (x, y, w, h)))
            else:
                classification_result.append(("Handwriting", text, conf, (x, y, w, h)))
    return classification_result

def main(pdf_path):
    images = pdf_to_images(pdf_path)
    all_classifications = []
    for image in images:
        processed_image = preprocess_image(image)
        classifications = classify_text(processed_image)
        all_classifications.append(classifications)
    return all_classifications

if __name__ == "__main__":
    pdf_path = 'sample.pdf'
    classifications = main(pdf_path)
    for page_num, page_classifications in enumerate(classifications):
        print(f"Page {page_num + 1}:")
        for classification in page_classifications:
            print(f"Type: {classification[0]}, Text: {classification[1]}, Confidence: {classification[2]}, Bounding Box: {classification[3]}")
