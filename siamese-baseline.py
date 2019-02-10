import os, sys
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")
SIZE = 224

class data_generator:
    def __init__(self, train_path, train_data, img_size=224, 
                 batch_size=32, augment=False, shuffle=True):
        self.img_size = img_size
        self.dim = (self.img_size, self.img_size, 3)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.data_info = self.read_info('train/', 'train.csv')
        self.pairs = self.create_data_pairs()

    def get_samp_size(self):
        return len(self.pairs)

    def read_info(self, path_to_train, data_path):
        # Load dataset info
        orig_data = pd.read_csv(data_path)
        I_dont_want_new_whales = orig_data['Id'] != 'new_whale'
        data = orig_data[I_dont_want_new_whales]
        unique_classes = pd.unique(data['Id'])
        encoding = dict(enumerate(unique_classes))
        decoding = {x:i for i,x in enumerate(unique_classes)}

        train_dataset_info = []
        for name, label in zip(data['Image'], data['Id']):
            train_dataset_info.append({
                'path':os.path.join(path_to_train, name),
                'label':int(decoding[label])})
        self.n_classes = len(pd.unique(data['Id']))
        return np.array(train_dataset_info)

    def create_data_pairs(self):
        return list(combinations(range(len(self.data_info)), 2))

    def create_train(self):
        if self.shuffle:
            np.random.shuffle(self.pairs)
        while True:
            for start in range(0, len(self.pairs), self.batch_size):
                end = min(start + self.batch_size, len(self.pairs))
                batch_image_pair1 = []
                batch_image_pair2 = []
                batch_pairs = self.pairs[start:end]
                batch_labels = np.zeros((len(batch_pairs),1))
                for i in range(len(batch_pairs)):
                    image1 = self.load_image(
                        self.data_info[batch_pairs[i][0]]['path'])
                    image2 = self.load_image(
                        self.data_info[batch_pairs[i][1]]['path'])
                    if self.augment:
                        image1 = self.image_augment(image1)
                        image2 = self.image_augment(image2)
                    batch_image_pair1.append(image1/255.)
                    batch_image_pair2.append(image2/255.)
                    if self.data_info[batch_pairs[i][0]]['label'] == self.data_info[batch_pairs[i][1]]['path']:
                        batch_labels[i] = 1
                    else:
                        batch_labels[i] = 0
                yield [np.array(batch_image_pair1, np.float32),np.array(batch_image_pair2, np.float32)], batch_labels

    def load_image(self, path):
        image = np.array(Image.open(path).convert('RGB'))
        image = cv2.resize(image, (self.img_size, self.img_size))
        return image

    def image_augment(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

from keras.models import Sequential, load_model
from keras.layers import *
from keras.applications.densenet import DenseNet121
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.models import Model

def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build_model(input_shape, lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=input_shape)  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=input_shape)
    img_b = Input(shape=input_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=['acc'])
    return model


# create callbacks list
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

epochs = 100; batch_size = 16
checkpoint = ModelCheckpoint('ResNet34.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = False)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                                   verbose=1, mode='auto', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10)
callbacks_list = [checkpoint, early, reduceLROnPlat]

# create train and valid datagens
train_generator = data_generator(img_size=384, batch_size=batch_size, 
                                 augment=True, train_path='train/', 
                                 train_data='train.csv')

# warm up model
model = build_model(
    input_shape = (SIZE,SIZE,3),
    lr = 1e-4,
    l2 = 2e-4)


model.fit_generator(
    train_generator.create_train(),
    steps_per_epoch=train_generator.get_samp_size() // batch_size,
    #validation_data=next(validation_generator),
    #validation_steps=np.floor(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs, 
    verbose=1,
    callbacks=callbacks_list)

# Create submit
encoding[n_classes] = 'new_whale'
n_classes += 1
submit = pd.read_csv('sample_submission.csv')
predicted = []
model.load_weights('ResNet34.h5')
for name in tqdm(submit['Image']):
    path = os.path.join('test/', name)
    image = data_generator.load_image(path, (SIZE,SIZE,3))/255.
    score_predict = model.predict(image[np.newaxis])[0]
    new_whale_prob = 1.0 - max(score_predict)
    score_predict = np.append(score_predict, new_whale_prob)
    label_predict = np.arange(n_classes)[np.argsort(score_predict)[-5:]]
    label_predict = [encoding[x] for x in label_predict[::-1]]
    str_predict_label = '{} {} {} {} {}'.format(*label_predict)
    predicted.append(str_predict_label)

submit['Id'] = predicted
submit.to_csv('submit_ResNet34.csv', index=False)
