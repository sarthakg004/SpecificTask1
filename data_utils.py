import pandas
import fitz
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


## Creating a fuction to convert pdf to images and split if double page

def pdf_to_images(pdf_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_doc = fitz.open(pdf_path)
    image_counter = 1 

    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        pix = page.get_pixmap()  # Render page as image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # If the page is significantly wider than tall, assume it's a double-page
        if img.width > img.height * 1.2:
            left = img.crop((0, 0, img.width // 2, img.height))  # Left half
            right = img.crop((img.width // 2, 0, img.width, img.height))  # Right half

            left.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")
            image_counter += 1
            right.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")
        else:
            img.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")

        image_counter += 1  # Increment for next image

    print(f"Processing complete! Saved {image_counter - 1} images in '{output_dir}'.")
    

def load_image(image_path):
    image = cv2.imread(image_path)  # Load image (default is BGR)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

def convert_to_grayscale(image):
    """Converts an image to grayscale if it's not already."""
    if len(image.shape) == 3:  # Check if the image is in color (BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already grayscale

def correct_skew(image):
    """Corrects skew in an image, supporting both grayscale and RGB inputs."""
    # Store original color state
    is_color = len(image.shape) == 3
    
    # Create grayscale copy for skew detection
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No text found, skipping skew correction.")
        return image  # Return original image with 0° skew angle

    # Find the largest contour (assumed to be text block)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area bounding box
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Normalize the angle to be between -45° and +45°
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    # Get image dimensions and center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation with border replication
    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed

def normalize_image(image):
    # Check if the image is RGB
    if len(image.shape) == 3:
        r, g, b = cv2.split(image)
        r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
        g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
        normalized = cv2.merge([r_norm, g_norm, b_norm])
        return normalized
    else:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
def ensure_300ppi(image, target_dpi=300):
    
    height, width = image.shape[:2]

    # Assume A4 document size in inches (common for scanned books)
    a4_width = 8.27  # inches
    a4_height = 11.69  # inches

    dpi_x = width / a4_width
    dpi_y = height / a4_height
    
    # Convert to PIL image, preserving color if needed
    image_pil = Image.fromarray(image)

    if dpi_x < target_dpi or dpi_y < target_dpi:
        # Calculate upscale factor
        scale_factor = target_dpi / min(dpi_x, dpi_y)  # Scale based on the lower DPI

        # Compute new size
        new_size = (int(image_pil.width * scale_factor), int(image_pil.height * scale_factor))

        # Resize using high-quality Lanczos resampling
        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)

        # Set the new DPI metadata
        image_pil.info['dpi'] = (target_dpi, target_dpi)

    # Convert back to OpenCV format
    image_upscaled = np.array(image_pil)

    return image_upscaled


def resize_and_pad(img, target_height, target_width, output_dir=None):
    # Calculate aspect ratio
    aspect = img.width / img.height

    # Determine new dimensions that fit within target size while preserving aspect ratio
    if aspect > (target_width / target_height):  # wider than target
        new_width = target_width
        new_height = int(new_width / aspect)
    else:  # taller than target
        new_height = target_height
        new_width = int(new_height * aspect)

    # Resize using LANCZOS for high quality
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create new white image of target size
    padded_img = Image.new("RGB", (target_width, target_height), color="white")

    # Center the resized image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))

    # Save the padded image
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(img.filename))
        padded_img.save(output_path)

    return padded_img