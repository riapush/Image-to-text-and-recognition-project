from PIL import Image
import pytesseract
import cv2
import os
import csv
import glob

def process_image(image_path):
    preprocess = "thresh"

    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

    # Save the temporary grayscale image
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # Apply OCR to the grayscale image
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)

    return text

def process_folder(folder_path, review, csv_writer):
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))

    for image_path in image_paths:
        try:
            text = process_image(image_path).replace('”', "'").replace('’', "'").replace('‘', "'").replace('™', "").replace('“', '')

            # Write the processed text, review, and folder name to CSV
            csv_writer.writerow([text, review])
        except Exception as e:
            print("Error processing image:", image_path)
            print(e)

with open('processed_images.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['review', 'sentiment'])

    positive_folder = os.path.join('test', 'positive')
    process_folder(positive_folder, 'positive', csv_writer)

    negative_folder = os.path.join('test', 'negative')
    process_folder(negative_folder, 'negative', csv_writer)

csvfile.close()
