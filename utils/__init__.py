import cv2
from cv2 import *
import numpy


def show(*show_image):
    print("showing " + str(len(show_image)))
    for i in range(len(show_image)):
        cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
    for i in range(len(show_image)):
        cv2.imshow(str(i), show_image[i])
    cv2.waitKey(0)


def draw_show(image, x1, y1, x2, y2):
    paint = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(paint, (y1, x1), (y2, x2), (0, 0, 255), 2)
    show(paint)


def fill_whole(image):
    image_copy = image.copy()
    x = len(image)
    y = len(image[0])
    mask = numpy.zeros([x+2,y+2], numpy.uint8)
    cv2.floodFill(image, mask, (0,0), 255)
    inv = cv2.bitwise_not(image)
    result = cv2.add(image_copy, inv)
    return result