import cv2
import numpy as np
import os
import glob
from preprocess_digit import process_image_to_mnist

def find_contours(image_path):
    os.makedirs("roi", exist_ok=True)
    for file in glob.glob("roi/*.png"):
        os.remove(file)
        print(f"Removed file: {file}")

    image = cv2.imread(image_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(image_grey, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Binary image', image_grey)

    # detect contours 
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    sorted_ctrs = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get boundoing box
        x, y, w, h = cv2.boundingRect(ctr)

        # Get ROI (region of interest)
        roi = image[y:y + h, x:x + w]

        # save only the ROI's which contain a valid information
        if h > 50 and w > 100:
            # show ROI
            cv2.imshow('segment no:'+str(i),roi)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.imwrite('roi\\{}.png'.format(i), roi)
    
    cv2.imshow('marked areas', image)

    # draw contours
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('Contour approximation', image_copy)
    cv2.waitKey(0)
    cv2.imwrite('contours_none_image1.png', image_copy)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if not len(sys.argv) == 2:
        print(f"Usage: python opencv.py <path_to_image>")
    else:
        find_contours(sys.argv[1])