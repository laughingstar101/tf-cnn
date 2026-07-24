import numpy as np
from PIL import Image, ImageFilter

def otsu_threshold(hist):
    """Return the Otsu threshold for a histogram (array of length 256)."""
    total_pixels = sum(hist)
    if total_pixels == 0:
        return 0

    sum_total = sum(i * hist[i] for i in range(256))
    sum_back = 0
    w_back = 0
    w_fore = 0
    var_max = 0
    threshold = 0

    for t in range(256):
        w_back += hist[t]
        if w_back == 0:
            continue
        w_fore = total_pixels - w_back
        if w_fore == 0:
            break
        sum_back += t * hist[t]
        mean_back = sum_back / w_back
        mean_fore = (sum_total - sum_back) / w_fore
        var_between = w_back * w_fore * (mean_back - mean_fore) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return threshold

def process_image_to_mnist(image_path, mode=0):
    # 1. Open and convert to grayscale
    im = Image.open(image_path).convert('L')
    img_arr = np.array(im, dtype=np.uint8)

    # 2. Compute histogram and Otsu threshold
    hist = np.histogram(img_arr, bins=256, range=(0, 255))[0]
    thresh = otsu_threshold(hist)

    # 3. Apply threshold: foreground = 255, background = 0
    binary = np.where(img_arr > thresh, 255, 0).astype(np.uint8)

    # 4. Ensure the digit is white (foreground) and background black
    # If more than half the pixels are white, we've inverted (background is white)
    binary = 255 - binary

    # 5. Find bounding box of the digit (non-zero pixels)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if not rows.any() or not cols.any():
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cropped = binary[y_min:y_max+1, x_min:x_max+1]

    # 6. Pad to square (preserve aspect ratio)
    h, w = cropped.shape
    side = max(h, w)
    pad_h = (side - h) // 2
    pad_w = (side - w) // 2
    square = np.pad(
        cropped,
        ((pad_h, side - h - pad_h), (pad_w, side - w - pad_w)),
        mode='constant',
        constant_values=0
    )

    # 7. Resize to 20x20
    pil_img = Image.fromarray(square)
    pil_img = pil_img.resize((20, 20), Image.Resampling.LANCZOS)
    resized = np.array(pil_img, dtype=np.float32)

    # 8. Normalize to 0-1 range
    if resized.max() > 0:
        resized = resized / 255.0

    # 9. Place onto 28x28 black canvas (center at 4,4)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = resized

    if mode == 0:
        return canvas.reshape(1, 28, 28, 1)
    else:
        out_img = (canvas * 255).astype(np.uint8)
        Image.fromarray(out_img, mode='L').save("processed_output.png")
        return "processed_output.png"

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess_digit.py <path_to_image>")
    else:
        processed = process_image_to_mnist(sys.argv[1])
        print(f"Processed image shape: {processed.shape}")
        print("Pixel value range:", processed.min(), "to", processed.max())

        # Save preview
        from PIL import Image
        out_img = (processed[0, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(out_img, mode='L').save("processed_output.png")
        print("Saved preview as 'processed_output.png'")