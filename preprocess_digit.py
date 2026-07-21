import numpy as np
from PIL import Image, ImageOps

def process_image_to_mnist(image_path, invert=False):
    """
    Load an image and preprocess it to match MNIST format.
    
    Args:
        image_path (str): Path to the input image (JPG, PNG, etc.).
        invert (bool): If True, force color inversion. If False, auto-detect.
    
    Returns:
        np.ndarray: Preprocessed image of shape (1, 28, 28, 1) with dtype float32,
                    ready to be fed into the TensorFlow model.
    """
    # 1. Open image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img, dtype=np.float32)

    # 2. Auto-invert if needed (unless forced by the user)
    if not invert:
        if np.mean(img_arr) < 128:
            img_arr = 255 - img_arr
    else:
        img_arr = 255 - img_arr

    # Convert back to PIL for easier manipulation
    img = Image.fromarray(img_arr.astype(np.uint8))

    # 3. Find the bounding box of the digit (pixels that are not white)
    arr = np.array(img)
    coords = np.argwhere(arr < 200)

    if coords.size == 0:
        # Fallback: if nothing found (e.g., image is empty), keep the whole image
        y_min, x_min = 0, 0
        y_max, x_max = arr.shape[0], arr.shape[1]
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1

    # Crop the digit
    cropped = arr[y_min:y_max, x_min:x_max]
    img_crop = Image.fromarray(cropped.astype(np.uint8))

    # 4. Add a small padding (15% of the larger side) so the digit doesn't touch the edges
    h, w = cropped.shape
    pad = int(max(h, w) * 0.15)
    img_crop = ImageOps.expand(img_crop, border=pad, fill=255)

    # 5. Resize while preserving aspect ratio to fit inside a 20x20 box
    w, h = img_crop.size
    max_side = max(w, h)
    scale = 20.0 / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6. Place the 20x20 (or smaller) image onto the center of a 28x28 white canvas
    canvas = Image.new('L', (28, 28), color=255)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas.paste(img_resized, (x_offset, y_offset))

    # 7. Convert to numpy array, normalize to [0,1], and reshape for the model
    final_arr = np.array(canvas, dtype=np.float32) / 255.0
    final_arr = final_arr.reshape(1, 28, 28, 1)

    return final_arr


if __name__ == "__main__":
    # Quick test: run this script with an image path as argument
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess_digit.py <path_to_image>")
    else:
        processed = process_image_to_mnist(sys.argv[1])
        print(f"Processed image shape: {processed.shape}")
        print("Pixel value range:", processed.min(), "to", processed.max())
        
        # (Optional) Save the processed image to verify visually
        from PIL import Image
        out_img = (processed[0, :, :, 0] * 255).astype(np.uint8)
        Image.fromarray(out_img, mode='L').save("processed_output.png")
        print("Saved preview as 'processed_output.png'")