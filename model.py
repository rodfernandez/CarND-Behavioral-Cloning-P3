import csv
import gc
import os
import sys

import cv2
import numpy as np
from keras.activations import elu
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential
from keras.objectives import mean_squared_error
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

BATCH_SIZE = 128
BRIGHTNESS_RANGE = -128, 0
CONTRAST_RANGE = -128, 128
EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_BRIGHTNESS_COUNT = 6
RANDOM_CONTRAST_COUNT = 0
STEERING_ANGLE_OFFSET = 0.25
STEERING_ANGLE_THRESHOLD = 0.001
VALIDATION_SPLIT = 0.2


def augment_data(samples, include_left=False, include_right=False, include_flipped=False, random_brightness=0,
                 random_contrast=0):
    augmented = samples.copy()

    if include_left:
        for path, steering_angle, *remainder in samples:
            if abs(steering_angle) >= STEERING_ANGLE_THRESHOLD and steering_angle + STEERING_ANGLE_OFFSET < 1:
                path = path.replace("IMG/center", "IMG/left")
                augmented.append((path, steering_angle + STEERING_ANGLE_OFFSET, False, 0, 0))

    if include_right:
        for path, steering_angle, *remainder in samples:
            if abs(steering_angle) >= STEERING_ANGLE_THRESHOLD and steering_angle - STEERING_ANGLE_OFFSET > -1:
                path = path.replace("IMG/center", "IMG/right")
                augmented.append((path, steering_angle - STEERING_ANGLE_OFFSET, False, 0, 0))

    if include_flipped:
        for path, steering_angle, *remainder in augmented.copy():
            if abs(steering_angle) >= STEERING_ANGLE_THRESHOLD:
                augmented.append((path, -steering_angle, True, 0, 0))

    augmented_copy = augmented.copy()

    if random_brightness > 0:
        for path, steering_angle, flip_it, *remainder in augmented_copy:
            if abs(steering_angle) >= STEERING_ANGLE_THRESHOLD:
                for brightness in np.random.randint(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1], random_brightness):
                    augmented.append((path, steering_angle, flip_it, brightness, 0))

    if random_contrast > 0:
        for path, steering_angle, flip_it, *remainder in augmented_copy:
            if abs(steering_angle) >= STEERING_ANGLE_THRESHOLD:
                for contrast in np.random.randint(CONTRAST_RANGE[0], CONTRAST_RANGE[1], random_contrast):
                    augmented.append((path, steering_angle, flip_it, 0, contrast))

    return shuffle(augmented)


def change_image_brightness(image, brightness):
    return np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)


def change_image_contrast(image, contrast):
    # See: https://goo.gl/gYKXAD
    factor = 259 * (contrast + 255) / (255 * (259 - contrast))
    return np.clip(factor * (image.astype(np.float) - 128) + 128, 0, 255).astype(np.uint8)


def create_model():
    model = Sequential()

    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation=elu))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation=elu))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation=elu))
    model.add(Conv2D(64, 3, 3, subsample=(2, 2), activation=elu))
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation=elu))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation=elu))
    model.add(Dense(512, activation=elu))
    model.add(Dense(128, activation=elu))
    model.add(Dense(16, activation=elu))
    model.add(Dense(1))

    model.compile(loss=mean_squared_error, optimizer=Adam(lr=LEARNING_RATE))

    return model


def create_generator(samples, batch_size=32):
    while 1:
        samples = shuffle(samples)

        for offset in range(0, len(samples), batch_size):
            batch = samples[offset:offset + batch_size]

            images = []
            angles = []

            for path, angle, flip_it, brightness, contrast in batch:
                image = cv2.imread(path, cv2.IMREAD_COLOR)

                if flip_it:
                    image = flip_image(image)

                if abs(brightness) > 0:
                    image = change_image_brightness(image, brightness)

                if abs(contrast) > 0:
                    image = change_image_contrast(image, contrast)

                images.append(image)
                angles.append(angle)

            yield shuffle(np.array(images), np.array(angles))


def flip_image(image):
    return cv2.flip(image, flipCode=1)


def load_data(log_path):
    data_dir = os.path.dirname(log_path)
    samples = []

    with open(log_path, 'r') as f:
        reader = csv.reader(f)

        for center_path, left_path, right_path, steering_angle, throttle, brake, speed in reader:
            sample = os.path.join(data_dir, center_path), float(steering_angle), False, 0, 0
            samples.append(sample)

    return samples


def plot_model(model):
    try:
        from keras.utils.visualize_util import plot
        plot(model, show_shapes=True)
    except:
        pass


if __name__ == "__main__":
    dataset = load_data(sys.argv[1])
    augmented = augment_data(dataset, True, True, True, RANDOM_BRIGHTNESS_COUNT, RANDOM_CONTRAST_COUNT)
    train_samples, validation_samples = train_test_split(augmented, test_size=VALIDATION_SPLIT)
    train_samples_count = len(train_samples)
    train_samples_generator = create_generator(train_samples, BATCH_SIZE)
    validation_samples_count = len(validation_samples)
    validation_samples_generator = create_generator(validation_samples, BATCH_SIZE)

    model = create_model()

    plot_model(model)

    model.fit_generator(generator=train_samples_generator,
                        samples_per_epoch=train_samples_count,
                        validation_data=validation_samples_generator,
                        nb_val_samples=validation_samples_count,
                        nb_epoch=EPOCHS)

    model.save('model.h5')

    gc.collect()
