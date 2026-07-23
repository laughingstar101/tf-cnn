import numpy as np
from PIL import Image, ImageFilter

def process_image_to_mnist(image_path):
    # 1. Open and convert to grayscale
    im = Image.open(image_path).convert('L')
    img_arr = np.array(im, dtype=np.uint8)

    # 2. Invert if necessary (so digit is white, background is black)
    if np.mean(img_arr) > 128:
        img_arr = 255 - img_arr

    # 3. Find the bounding box of the actual digit (non-zero pixels)
    rows = np.any(img_arr, axis=1)
    cols = np.any(img_arr, axis=0)
    
    # If the image is completely blank, return a zero array
    if not rows.any() or not cols.any():
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 4. Crop the digit tightly
    cropped = img_arr[y_min:y_max+1, x_min:x_max+1]

    # 5. Pad to a square (to avoid distortion when resizing)
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

    # 6. Resize to 20x20 using high-quality LANCZOS
    pil_img = Image.fromarray(square)
    pil_img = pil_img.resize((20, 20), Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)
    resized = np.array(pil_img, dtype=np.float32)

    # 7. Normalize to 0-1 range
    if resized.max() > 0:
        resized = resized / 255.0

    # 8. Place onto a 28x28 black canvas (center at 4,4)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[4:24, 4:24] = resized

    # 9. Return as (1, 28, 28, 1)
    return canvas.reshape(1, 28, 28, 1)


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