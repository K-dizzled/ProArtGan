from PIL import Image
import numpy as np
import os
import sys


def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    return pixels


def resize_landscape(image, required_size=(256, 256)):
    # If image is high enough, we can split it into 2 images
    if image.shape[0] >= image.shape[1] * 2:
        middle = int(image.shape[0] / 2)
        image1 = image[:middle, :, :]
        image2 = image[middle:, :, :]
        # Resize images
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        image1 = image1.resize(required_size)
        image2 = image2.resize(required_size)

        return np.asarray(image1), np.asarray(image2)

    # If image is wide enough, we can split it into 2 images
    if image.shape[1] >= image.shape[0] * 2:
        middle = int(image.shape[1] / 2)
        image1 = image[:, :middle, :]
        image2 = image[:, middle:, :]
        # Resize images
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        image1 = image1.resize(required_size)
        image2 = image2.resize(required_size)

        return np.asarray(image1), np.asarray(image2)

    image = Image.fromarray(image)
    image = image.resize(required_size)
    return np.asarray(image), None


def load_images(directory):
    landscapes = list()
    for i, filename in enumerate(os.listdir(directory)):
        print(f'Image number {i}')
        # Load the image
        image = load_image(directory + filename)
        # Resize
        resized = resize_landscape(image)
        # Store
        if resized[1] is not None:
            landscapes.append(resized[1])
        landscapes.append(resized[0])

    return np.asarray(landscapes)


data_dir = '/Users/andrei/PycharmProjects/DL_Book_Manning/Art_Gan/landscapes_dataset/'
images = load_images(data_dir)
np.savez_compressed('landscapes_256by256_dataset.npz', images)
