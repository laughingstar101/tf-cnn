import cv2
import numpy as np
import os
import glob
from preprocess_digit import process_image_to_mnist

def sort_contours(contours):
    """Sort contours: cluster rows by center-y, then sort each row left-to-right."""
    if not contours:
        return [], []

    # Get bounding boxes and heights
    boxes = [cv2.boundingRect(c) for c in contours]
    heights = [h for (x, y, w, h) in boxes]
    median_height = np.median(heights) if heights else 28

    # Sort by center_y
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]/2)

    rows = []
    current_row = []
    if not sorted_contours:
        return [], []

    # Start with the first contour's center_y as the row reference
    first_c = sorted_contours[0]
    current_center_y = cv2.boundingRect(first_c)[1] + cv2.boundingRect(first_c)[3]/2

    for c in sorted_contours:
        x, y, w, h = cv2.boundingRect(c)
        center_y = y + h/2
        if abs(center_y - current_center_y) < 1.2 * median_height:
            current_row.append(c)
        else:
            # Different row: sort current row by x and add
            rows.append(sorted(current_row, key=lambda c2: cv2.boundingRect(c2)[0]))
            current_row = [c]
            current_center_y = center_y

    if current_row:
        rows.append(sorted(current_row, key=lambda c2: cv2.boundingRect(c2)[0]))

    sorted_ctrs = [c for row in rows for c in row]
    return sorted_ctrs, rows

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

    TARGET_MIN_DIM = 3000
    TARGET_MAX_DIM = 3000
    h, w = image.shape[:2]
    max_side = max(h, w)
    
    if max_side > TARGET_MAX_DIM:
        scale = TARGET_MAX_DIM / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if args.debug:
            print(f"[DEBUG] Downscaled from {w}x{h} to {new_w}x{new_h}")
    elif max_side < TARGET_MIN_DIM:
        scale = TARGET_MIN_DIM / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        if args.debug:
            print(f"[DEBUG] Upscaled from {w}x{h} to {new_w}x{new_h}")
    else:
        if args.debug:
            print(f"[DEBUG] Image size {w}x{h} is within target range, no resizing")

    # Denoise
    image = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

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
    cv2.imwrite('debug/_binary.png', binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs, rows = sort_contours(contours)

    # Dynamic row colours
    num_rows = len(rows)
    # Generate distinct colours using HSV
    row_colors = []
    for i in range(num_rows):
        hue = i * (180 / max(num_rows, 1))  # 0–180 hue range
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
        row_colors.append(tuple(color))

    rows_image = image.copy()
    for row_idx, row in enumerate(rows):
        color = row_colors[row_idx % len(row_colors)]
        for c in row:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(rows_image, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite('debug/_rows.png', rows_image)
    if args.debug:
        cv2.imshow('Rows', rows_image)

    marked_image = image.copy()

    # Dynamic minimum area
    min_area = min(100, 0.001 * image.shape[0] * image.shape[1])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        area = cv2.contourArea(ctr)
        if area < min_area:
            continue

        contour_mask = np.zeros_like(binary)
        cv2.drawContours(contour_mask, [ctr], -1, 255, -1)

        mask_cropped = contour_mask[y:y + h, x:x + w]

        roi = binary[y:y + h, x:x + w]
        roi_cleaned = cv2.bitwise_and(roi, roi, mask=mask_cropped)

        if args.debug:
            cv2.imshow(f'ROI_{i}', roi_cleaned)
        cv2.imwrite(f'roi/{i}.png', roi_cleaned)
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if args.debug:
        cv2.imshow('Marked areas', marked_image)
    cv2.imwrite('debug/_marked_image.png', marked_image)

    # Draw all contours (debugging)
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    if args.debug:
        cv2.imshow('Contour approximation', contours_image)
    cv2.imwrite('debug/_contours_image.png', contours_image)

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