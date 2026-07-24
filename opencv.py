import cv2
import numpy as np
import os
import glob
from preprocess_digit import process_image_to_mnist

def find_contrast_channel(image):
    """Return the grayscale image from the channel with highest contrast."""
    if len(image.shape) == 2:
        return image
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    stds = [np.std(ch) for ch in channels]
    best = np.argmax(stds)
    return channels[best]

def get_background_colour(image, border_ratio=0.05):
    """
    Sample pixels from the four borders to estimate the background colour.
    Returns the average LAB colour of the border region.
    """
    h, w = image.shape[:2]
    border_pixels = []
    border_h = int(h * border_ratio)
    border_w = int(w * border_ratio)
    # Top and bottom borders
    border_pixels.append(image[0:border_h, :].reshape(-1, 3))
    border_pixels.append(image[h-border_h:h, :].reshape(-1, 3))
    # Left and right borders (excluding corners already included)
    border_pixels.append(image[border_h:h-border_h, 0:border_w].reshape(-1, 3))
    border_pixels.append(image[border_h:h-border_h, w-border_w:w].reshape(-1, 3))
    border_pixels = np.vstack(border_pixels)
    # Average colour in LAB space
    lab = cv2.cvtColor(border_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    bg_lab = np.mean(lab, axis=0)
    return bg_lab

def find_contours(image_path, args):
    os.makedirs("roi", exist_ok=True)
    os.makedirs("debug", exist_ok=True)
    filename = os.path.basename(image_path)
    with open("roi/info.txt", "w") as f:
        f.write(filename)
    for file in glob.glob("roi/*.png"):
        os.remove(file)
        if args.debug:
            print(f"[DEBUG] Removed file: {file}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    # ---- Colour‑based segmentation ----
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bg_lab = get_background_colour(image)
    if args.debug:
        print(f"[DEBUG] Background LAB: {bg_lab}")

    diff = lab - bg_lab.astype(np.float32)
    dist = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.uint8)

    # Use percentile threshold instead of Otsu
    percentile = 60   # tune this (50–80) depending on your images
    thresh_val = np.percentile(dist, percentile)
    _, binary_colour = cv2.threshold(dist, thresh_val, 255, cv2.THRESH_BINARY)

    white_pixels = np.sum(binary_colour == 255)
    total_pixels = binary_colour.size
    if white_pixels > total_pixels // 2:
        binary_colour = cv2.bitwise_not(binary_colour)

    kernel = np.ones((5, 5), np.uint8)   # larger kernel
    binary_colour = cv2.morphologyEx(binary_colour, cv2.MORPH_CLOSE, kernel)
    binary_colour = cv2.morphologyEx(binary_colour, cv2.MORPH_OPEN, kernel)

    # Fallback if colour segmentation fails
    white_pixels = np.sum(binary_colour == 255)
    if white_pixels < 0.001 * total_pixels or white_pixels > 0.9 * total_pixels:
        if args.debug:
            print("[DEBUG] Colour segmentation poor, falling back to grayscale.")
        gray_original = find_contrast_channel(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray_original)
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_pixels = np.sum(binary_otsu == 255)
        total_pixels = binary_otsu.size

        if white_pixels == 0 or white_pixels > 0.9 * total_pixels:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 15, 10)
            white_pixels = np.sum(binary == 255)
            if white_pixels > total_pixels // 2:
                binary = cv2.bitwise_not(binary)
        else:
            if white_pixels > total_pixels // 2:
                binary = cv2.bitwise_not(binary_otsu)
            else:
                binary = binary_otsu

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        binary = binary_colour

    if args.debug:
        cv2.imshow('Binary', binary)
    cv2.imwrite('debug/binary.png', binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    marked_image = image.copy()

    # Dynamic minimum area
    min_area = max(100, 0.001 * image.shape[0] * image.shape[1])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        area = cv2.contourArea(ctr)
        if area < min_area:
            continue

        roi = image[y:y + h, x:x + w]
        if args.debug:
            cv2.imshow(f'ROI_{i}', roi)
        cv2.imwrite(f'roi/{i}.png', roi)
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if args.debug:
        cv2.imshow('Marked areas', marked_image)
    cv2.imwrite('debug/marked_image.png', marked_image)

    # Draw all contours (debugging)
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    if args.debug:
        cv2.imshow('Contour approximation', contours_image)
    cv2.imwrite('debug/contours_image.png', contours_image)

    if args.debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='path to image')
    parser.add_argument('--debug', action="store_true", dest='debug', help='enable debug')
    args = parser.parse_args()
    find_contours(args.image, args)