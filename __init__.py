from enum import unique
from posixpath import split
from typing import Dict
from wsgiref import validate
import matplotlib.pyplot as plt
from sympy import beta, subsets
import numpy as np
import os
import PIL
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model

from keras import backend as K

from tensorflow import keras
from keras import layers
import pandas as pd
import pathlib
import glob


train_ds = None
validate_ds = None
test_ds = None
img_height = 28
img_width = 28


def predict(model):
    print('beginning predictions')
    predictions = model.predict(test_ds.take(-1))
    predictions = sort(predictions)
    print('predictions complete')
    return predictions


def setupModel() -> Model:
    model = keras.models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),

        layers.Conv2D(32, 3, padding='same', activation='relu'),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Dropout(0.3),

        layers.MaxPooling2D(),

        layers.Dropout(0.1),
        layers.Conv2D(32, 3, padding='same', activation='relu'),

        layers.Dropout(0.1),
        layers.Conv2D(64, 3, padding='same', activation='relu'),

        layers.Dropout(0.3),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),

        layers.GlobalAveragePooling2D(),

        layers.Dense(32, activation='relu'),

        layers.Dense(8, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model


def train(epochs, model: Model):

    global train_ds
    global test_ds
    global validate_ds

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = validate_ds.cache().prefetch(buffer_size=AUTOTUNE)

    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        latest_accuracy = str(
            history.history['val_sparse_categorical_accuracy'][-1])

        print('Latest accuracy: ' + latest_accuracy)
    except KeyboardInterrupt:
        print('\nTraining interrupted - predicting now ...')

    predictions = predict(model)

    filename = str('(epochs - ' + str(epochs) + ') predictions')

    pd.DataFrame(data={'Predictions': predictions}).to_csv(
        'predictions.csv', index_label='Id')
    pd.DataFrame(data={'Predictions': predictions}).to_csv(
        'past_predictions/ ' + filename + '.csv', index_label='Id')

    try:
        model.save('past_models/' + filename + '.h5')
        print('Model saved as ' + filename + '.h5 in past_models/')
    except:
        print('Model not saved')
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


def loadData():
    global train_ds
    global test_ds
    global validate_ds
    # create a path object to the training data (images of kidneys)
    train_data_dir = pathlib.Path("./train_sorted/")
    test_data_dir = pathlib.Path("./test/")
    # count the number of images in the training data
    # list of all images in the training data
    # train_data = list(train_data_dir.glob('./train/*.jpg'))

    # Set the height and width of the images
    img_height = 28
    img_width = 28
    batch_size = 200

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        seed=123,
        validation_split=0.33,
        labels='inferred',
        subset='training',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
    )

    validate_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        seed=123,
        validation_split=0.33,
        labels='inferred',
        subset='validation',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        shuffle=False,
        labels=None,
        image_size=(img_height, img_width),
        batch_size=1,
        color_mode='grayscale'
    )

    print('data loaded')


def loadModel(filename) -> Model:
    model = keras.models.load_model(filename)
    print('model loaded')
    return model


def loadAndPredict(filename):
    model = loadModel(filename)
    predictions = predict(model)
    filename = 'predictions_for_model_' + str(filename)
    pd.DataFrame(data={'Cell type': predictions}).to_csv(
        'past_predictions/ ' + filename + '.csv', index_label='Id')
    return predictions


if (__name__ == "__main__"):
    # model = make_model('Models//NewIV3-150x150x1-3C.h5')
    # model = loadModel('past_models/(62.476%)_predictions.h5')
    model = setupModel()
    loadData()
    train(100, model)

    # model = setupModle()
    # loadAndPredict('Models//NewIV3-150x150x1-3C.h5')
    # train(130)
