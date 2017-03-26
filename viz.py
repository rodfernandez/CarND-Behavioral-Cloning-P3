import re
import sys
from random import choice

import cv2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from model import augment_data, change_image_brightness, change_image_contrast, flip_image, load_data

PARSE_OUTPUT_PATTERN = ' loss: ((?:\d|\.)+) - val_loss: ((?:\d|\.)+)'


def collate(images):
    height, width, channels = images[0].shape

    result = np.full((height, width * len(images), channels), 255, np.uint8)

    for i, image in zip(range(len(images)), images):
        result[0:height, i * width:i * width + width] = image[0:height, 0:width]

    return result


def parse_output(filename='./nohup.out'):
    train_loss = []
    validation_loss = []

    with open(filename, 'r') as f:
        for match in re.finditer(PARSE_OUTPUT_PATTERN, f.read()):
            train_loss.append(match.group(1))
            validation_loss.append(match.group(2))

    f.close()

    return train_loss, validation_loss


def save_brightness_image(image_path, filename='./images/brightness.png'):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    images = [image]

    for brightness in [-128, -64, 64, 128]:
        images.append(change_image_brightness(image, brightness))

    cv2.imwrite(filename, collate(images))


def save_contrast_image(image_path, filename='./images/contrast.png'):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    images = [image]

    for contrast in [-128, -64, 64, 128]:
        images.append(change_image_contrast(image, contrast))

    cv2.imwrite(filename, collate(images))


def save_cameras_image(center_path, filename='./images/cameras.png'):
    left_path = center_path.replace('IMG/center', 'IMG/left')
    right_path = center_path.replace('IMG/center', 'IMG/right')

    center_image = cv2.imread(center_path, cv2.IMREAD_COLOR)
    left_image = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_image = cv2.imread(right_path, cv2.IMREAD_COLOR)

    cv2.imwrite(filename, collate([left_image, center_image, right_image]))


def save_flipped_image(image_path, filename='./images/flipped.png'):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    flipped = flip_image(image)

    cv2.imwrite(filename, collate([image, flipped]))


def save_histogram(samples, filename='./images/hist.png'):
    x = list(map(lambda x: x[1], samples))

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x, bins=50, normed=True)
    mu, sigma = norm.fit(x)
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, 'r--')
    ax.set_xlim([-1, 1])
    ax.set_xlabel('Normalized steering angle')
    ax.set_ylabel('Probability density')

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_steering_image(samples, filename='./images/steering.png'):
    y = list(map(lambda x: x[1], samples))

    plt.plot(y)
    plt.ylabel('Normalized steering angle')
    plt.axis([0, len(y), -1, 1])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_training_loss(filename='./images/training_loss.png'):
    train_loss, validation_loss = parse_output()
    epochs = len(train_loss)
    x = range(1, epochs + 1, 1)

    fig, ax = plt.subplots()
    ax.plot(x, train_loss, 'ro-', label='Training')
    ax.plot(x, validation_loss, 'bs-', label='Validation')
    ax.axis([1, epochs, 0, plt.axis()[3]])
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    dataset = load_data(sys.argv[1])
    augmented = augment_data(dataset, True, True, True, 6, 0)

    save_steering_image(dataset)
    save_histogram(dataset, filename='./images/original_hist.png')
    save_histogram(augmented, filename='./images/augmented_hist.png')

    random_sample = choice(dataset)
    random_sample_image_path = random_sample[0]

    save_cameras_image(random_sample_image_path)
    save_flipped_image(random_sample_image_path)
    save_brightness_image(random_sample_image_path)
    save_contrast_image(random_sample_image_path)

    save_training_loss()
