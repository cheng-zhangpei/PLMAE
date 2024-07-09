"""
@Function:  train v1 mae model
@Author : ZhangPeiCheng
@Time : 2023/12/3 9:17
"""
import os
import keras.metrics
import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.utils.version_utils import callbacks

def train_model_with_spatial_info(epochs, batch_size, x1, x2, y, model, model_name, initial_learning_rate):
    tensorboard = callbacks.TensorBoard(histogram_freq=1)
    x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=30)
    x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size=0.3, random_state=30)
    # x3_train, x3_test, y_train, y_test = train_test_split(x3, y, test_size=0.3, random_state=30)
    print("train set shape")
    print(x1_train.shape)
    print(x2_train.shape)
    # print(x3_train.shape)
    print("label shape")
    print(y_train.shape)
    decay_steps = epochs * (x1_train.shape[0] / batch_size)
    decay_rate = 0.95
    early_stop = EarlyStopping(monitor='val_mae', min_delta=0.0001, patience=20, mode='min',
                               restore_best_weights=True)

    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=True)
    model.compile(
        optimizer=optimizers.Adam(learning_rate_fn),
        loss='mae',
        metrics=['mae']
    )
    his = model.fit([x1_train, x2_train],
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[early_stop],
                    validation_data=(
                        [x1_test, x2_test], y_test)
                    )
    save_path = "./trained_model/" + model_name
    model.save(save_path, save_format='tf', overwrite=True)
    # tf.keras.models.save_model(
    #     model,
    #     save_path,
    #     overwrite=True,
    #     save_format=None,
    #     signatures=None,
    #     options=None
    # )
    val_accuracy = his.history['val_mae']
    return val_accuracy
