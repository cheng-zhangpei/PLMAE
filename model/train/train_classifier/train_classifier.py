"""
@Function:             
@Author : ZhangPeiCheng
@Time : 2024/1/6 11:24
"""
import os
import keras.metrics
import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.utils.version_utils import callbacks

from model.classifier.classifier import classifier
from train_set_build.classifier_train_set_bulider import classifier_dataset_builder


def train_model_classifier(epochs, batch_size, x, y, model, model_name, initial_learning_rate):
    # 对y进行独热编码
    y = to_categorical(y, num_classes=2)
    print(y.shape)
    print(x.shape)
    x1_train, x1_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30)
    decay_steps = epochs * (x1_train.shape[0] / batch_size)
    decay_rate = 0.95
    early_stop = EarlyStopping(monitor='val_binary_accuracy', min_delta=0.0001, patience=15, mode='max',
                               restore_best_weights=True)

    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=True)
    model.compile(
        optimizer=optimizers.Adam(learning_rate_fn),
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy()]
    )
    model.fit(x1_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[early_stop],
                    validation_data=(
                        x1_test, y_test)
                    )
    save_path = "./trained_model/" + model_name
    model.save(save_path, save_format='tf', overwrite=True)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # 指定使用第一块GPU来进行训练
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    bit = 3
    dir_path = r"D:\czp\combineData"
    model_name = r"D:\czp\mae\trained_model\index_change_8"
    dataset,labels = classifier_dataset_builder(bit, dir_path, 8, model_name)
    model = classifier(dataset[0].shape)
    train_model_classifier(epochs=100,batch_size=100,x=dataset,y=labels,
                           model=model,model_name="classify_8",initial_learning_rate=5e-4)
