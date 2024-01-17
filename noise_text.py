from PIL import Image
import cv2
import numpy as np
import os
import glob
import pytesseract
import csv

def add_salt_pepper_noise(image):
    # Convert the image to a numpy array
    img = np.array(image)

    # Define the probability of salt and pepper noise
    p = 0.05

    # Generate random noise
    noisy_image = np.copy(img)
    noisy_pixels = np.random.rand(*img.shape[:2])
    noisy_image[noisy_pixels < p / 2] = 0  # Salt noise
    noisy_image[noisy_pixels > 1 - p / 2] = 255  # Pepper noise

    # Convert the noisy image back to PIL Image format
    noisy_image = Image.fromarray(noisy_image)

    return noisy_image

def process_image(image_path):

    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Save the temporary grayscale image
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # Apply OCR to the grayscale image
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)

    return text


image = Image.open("noisy.png")
text = process_image("noisy.png")
print("Extracted Text:\n", text)
