# Source - https://stackoverflow.com/a/55968472
# Posted by Naveen Kumar Dasari, modified by community. See post 'Timeline' for change history
# Retrieved 2026-07-22, License - CC BY-SA 4.0
# Modified to work with modern Pillow and return (1,28,28,1) array with 0-255 values.

import numpy as np
from PIL import Image, ImageFilter

def process_image_to_mnist(argv):
    """
    This function returns the pixel values as a numpy array of shape (1, 28, 28, 1)
    with values in range 0-255 (black digit on white background).
    The input is a png file location.
    """
    # 1. Open and convert to grayscale
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])

    # 2. Auto-invert if background is dark (mean < 128)
    img_arr = np.array(im, dtype=np.float32)
    if np.mean(img_arr) < 128:
        im = Image.fromarray((255 - img_arr).astype(np.uint8))

    # 3. Create white canvas of 28x28
    newImage = Image.new('L', (28, 28), 255)

    # 4. Resize to 20-pixel side while preserving aspect ratio, then paste at offset (4,4)
    if width > height:
        # Width is bigger → width becomes 20 pixels
        nheight = int(round((20.0 / width * height), 0))
        if nheight == 0:
            nheight = 1
        img = im.resize((20, nheight), Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        # Height is bigger → height becomes 20 pixels
        nwidth = int(round((20.0 / height * width), 0))
        if nwidth == 0:
            nwidth = 1
        img = im.resize((nwidth, 20), Image.Resampling.LANCZOS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    # 5. Convert to numpy array and reshape to (1, 28, 28, 1)
    arr = np.array(newImage, dtype=np.float32)   # shape (28, 28)
    arr = arr.reshape(1, 28, 28, 1)              # shape (1, 28, 28, 1)
    return arr


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess_digit.py <path_to_image>")
    else:
        processed = process_image_to_mnist(sys.argv[1])
        print(f"Processed image shape: {processed.shape}")
        print("Pixel value range:", processed.min(), "to", processed.max())

        # Save preview (no scaling needed, already 0-255)
        from PIL import Image
        out_img = processed[0, :, :, 0].astype(np.uint8)
        Image.fromarray(out_img, mode='L').save("processed_output.png")
        print("Saved preview as 'processed_output.png'")