import numpy as np
import scipy
from imageio import imread


def circle(size, radius, center=(125, 125)):
    """
    Create a circle of a certain radius
    Arguments:
        size (int): size of the array
        radius (int): radius of the circle to be drawn
        center (tuple of ints): row and column of the array where the circle will be centered
    Returns:
        A 2d array filled in with ones where the zero is
    """
    arr = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius ** 2):
                arr[i, j] = 1
    return arr


def square(size, side, angle=0):
    """
    Creates a square with an angle of rotation
    Arguments:
        size (int): Size of the array
        side (int): length of the side of the square
        angle (float): angle of rotation in radians
    Returns:
        An array with a centered square
    """
    center = int(size / 2)
    half_side = int(side / 2)
    arr = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i >= center - half_side) and (i <= center + half_side) and (j >= center - half_side) and (
                    j <= center + half_side):
                arr[i, j] = 1
    return scipy.ndimage.rotate(arr, angle, reshape=False)


def cross(size, side, angle=0, width=5):
    """
    Creates a cross rotated by a given angle
    Arguments:
        size (int): size of the array
        side (int): length of the cross
        angle (float): angle of rotation in radians
        width (int): width of the cross
    Returns
        An array with a centered cross
    """
    center = int(size / 2)
    half_side = int(side / 2)
    arr = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i >= center - half_side) and (i <= center + half_side) and (j >= center - half_side) and (
                    j <= center + half_side) and ((center - width <= i <= center + width) or (center - width <= j <= center + width)):
                arr[i, j] = 1
    return scipy.ndimage.rotate(arr, angle, reshape=False)


def image_from_file(path):
    """
    Reads a file and returns an array from 0 to 1 based on the images' values
    """
    if path[-3:] == "npy":
        arr = np.load(path)
        # Take a mean of the RGB values
        if len(arr.shape > 2):
            arr = np.mean(arr, 2)
        max_val = np.max(arr)
        arr = arr / max_val
    else:
        arr = imread(path)
        # Take a mean of the RGB values
        arr = np.mean(arr, 2)
        max_val = np.max(arr)
        arr = arr / max_val
    return arr


def read_image_array(path):
    """
    Reads a numpy array of W x H x T where T are either subsequent time frames or different images
    """
    arr = np.load(path)
    arr = arr / np.mean(arr, axis=(1, 2))
    return arr
