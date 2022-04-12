from enum import unique
from posixpath import split
from typing import Dict
import matplotlib.pyplot as plt
from sympy import beta, subsets
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
import pandas as pd
import pathlib
import glob


def main():
    # read in the training mappings id to kidney type
    train_dict = pd.read_csv('train.csv')

    # create a path object to the training data (images of kidneys)
    train_data_dir = pathlib.Path("./train_sorted/")
    test_data_dir = pathlib.Path("./test/")
    # count the number of images in the training data
    # list of all images in the training data
    # train_data = list(train_data_dir.glob('./train/*.jpg'))

    # Set the height and width of the images
    img_height = 28
    img_width = 28
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        seed=123,
        validation_split=0.2,
        labels='inferred',
        subset='training',
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    validate_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        seed=123,
        validation_split=0.2,
        labels='inferred',
        subset='validation',
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        shuffle=True,
        labels=None,
        image_size=(img_height, img_width),
        batch_size=1
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = validate_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = keras.models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(8)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    epochs = 100
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    predictions = model.predict(test_ds.take(-1))
    predictions = sort(predictions)
    latest_accuracy = history.history['val_accuracy'][-1]
    print(latest_accuracy)
    pd.DataFrame(data={'Predictions': predictions}).to_csv(
        'predictions.csv', index_label='Id')
    return


def sort(predictions):

    a = np.argmax(predictions, axis=1)
    b = [None] * 50000

    # Iterate through the predictions and change the value to the name of the kidney type
    it = 0
    for i in a:
        if i == 0:
            b[it] = 0
        elif i == 1:
            b[it] = 1
        elif i == 2:
            b[it] = 2
        elif i == 3:
            b[it] = 3
        elif i == 4:
            b[it] = 4
        elif i == 5:
            b[it] = 5
        elif i == 6:
            b[it] = 6
        elif i == 7:
            b[it] = 7
        elif i == 8:
            b[it] = 8
        it = it + 1

    return b


if (__name__ == "__main__"):
    main()
