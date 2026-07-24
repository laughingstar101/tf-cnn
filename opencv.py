import cv2
import numpy as np
import os
import glob
from preprocess_digit import process_image_to_mnist

def find_contours(image_path, args):
    os.makedirs("roi", exist_ok=True)
    for file in glob.glob("roi/*.png"):
        os.remove(file)
        print(f"Removed file: {file}")

    image = cv2.imread(image_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(image_grey, 127, 255, cv2.THRESH_BINARY_INV)
    if args.debug: cv2.imshow('Binary image', image_grey)

    # detect contours 
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    sorted_ctrs = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    marked_image = image.copy()

    for i, ctr in enumerate(sorted_ctrs):
        # Get boundoing box
        x, y, w, h = cv2.boundingRect(ctr)

        area = cv2.contourArea(ctr)
        if area < 100:
            continue

        # Get ROI (region of interest)
        roi = image[y:y + h, x:x + w]

        # save only the ROI's which contain a valid information
        if args.debug: cv2.imshow(f'{i}', roi)
        cv2.imwrite('roi\\{}.png'.format(i), roi)
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if args.debug: cv2.imshow('Marked areas', marked_image)

    # draw contours
    image_copy = image.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    if args.debug: cv2.imshow('Contour approximation', image_copy)
    cv2.waitKey(0)
    cv2.imwrite('contours_none_image1.png', image_copy)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='path to image')
    parser.add_argument('--debug', type=int, default=0, help='enable debug')
    args = parser.parse_args()
    find_contours(args.image, args)