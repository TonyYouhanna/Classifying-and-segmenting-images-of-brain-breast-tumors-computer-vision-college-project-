
import os
import cv2
from PIL import Image as im
from random import shuffle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
# from keras_segmentation.models.unet import vgg_unet
# from segmentation_models import Unet
# import segmentation_models as sm

from keras.metrics import MeanIoU, IoU, BinaryIoU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate
from tensorflow.keras import Sequential
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = "Dataset/"
IMG_SIZE = 224
Brain_Breast_Train = ["Brain scans/No tumor/Train", "Brain scans/Tumor/TRAIN", "Breast scans/benign/Train", "Breast scans/malignant/Train", "Breast scans/normal/Train"]
Brain_Breast_Test = ["Brain scans/No tumor/Test", "Brain scans/Tumor/TEST", "Breast scans/benign/Test", "Breast scans/malignant/Test", "Breast scans/normal/Test"]
Brain_Train = ["Brain scans/No tumor/Train", "Brain scans/Tumor/TRAIN"]
Brain_Tumor_Train = ["Brain scans/Tumor/TRAIN", "Brain scans/Tumor/TRAIN_masks"]
Brain_Tumor_Test = ["Brain scans/Tumor/TEST", "Brain scans/Tumor/TEST_masks"]
Breast_Benign_Train = "Breast scans/benign/Train"
Breast_Malignant_Train = "Breast scans/malignant/Train"
Breast_Benign_Test = "Breast scans/benign/Test"
Breast_Malignant_Test = "Breast scans/malignant/Test"
Brain_Test = ["Brain scans/No tumor/Test", "Brain scans/Tumor/TEST"]
Breast_Train = ["Breast scans/benign/Train", "Breast scans/malignant/Train", "Breast scans/normal/Train"]
Breast_Test = ["Breast scans/benign/Test", "Breast scans/malignant/Test", "Breast scans/normal/Test"]
def create_Brain_Breast_train_data():
  training_data = []
  for i in range(len(Brain_Breast_Train)):
    path = os.path.join(ROOT, Brain_Breast_Train[i])
    for img in os.listdir(path):
      if img.find('_mask') == -1:
        Full_path = os.path.join(path, img)
        img_data = cv2.imread(Full_path)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        if 'Brain scans' in Full_path:
          training_data.append([img_data, 0])
        else:
          training_data.append([img_data, 1])
  shuffle(training_data)
  np.save('brain_breast_training_data.npy', training_data)
  return training_data


if os.path.exists('brain_breast_training_data.npy'):
  Brain_Breast_train_data = np.load('brain_breast_training_data.npy', allow_pickle=True)
else:
  Brain_Breast_train_data = create_Brain_Breast_train_data()

BB_X_train = np.array([i[0] for i in Brain_Breast_train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
BB_y_train = [i[1] for i in Brain_Breast_train_data]
BB_Y_train_OHE = [[int(j == i) for j in range(max(BB_y_train) + 1)] for i in BB_y_train]
BB_Y_train_reshaped = np.array(BB_Y_train_OHE)
BB_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, rescale = 1.0/255.)
BB_datagen.fit(BB_X_train)

if os.path.exists('Brain_Breast_VGG19.h5'):
    Brain_Breast_VGG19 = load_model('Brain_Breast_VGG19.h5')
else:
  Brain_Breast_VGG19 = Sequential()
  Brain_Breast_VGG19.add(VGG19(input_shape = (IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet'))
  for each_layer in Brain_Breast_VGG19.layers:
    each_layer.trainable = False
  Brain_Breast_VGG19.add(Flatten())
  Brain_Breast_VGG19.add(Dense(2, activation='softmax'))
  Brain_Breast_VGG19.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), metrics = ['accuracy'])
  Brain_Breast_VGG19.summary()
  Brain_Breast_VGG19.fit(BB_datagen.flow(BB_X_train, BB_Y_train_reshaped), epochs=2, verbose='auto')
  Brain_Breast_VGG19.save('Brain_Breast_VGG19.h5')

def create_Brain_train_data():
  training_data = []
  for i in range(len(Brain_Train)):
    path = os.path.join(ROOT, Brain_Train[i])
    for img in os.listdir(path):
      if img.find('_mask') == -1:
        Full_path = os.path.join(path, img)
        img_data = cv2.imread(Full_path)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        if 'No tumor' in Full_path:
          training_data.append([img_data, 0])
        else:
          training_data.append([img_data, 1])
  shuffle(training_data)
  np.save('brain_training_data.npy', training_data)
  return training_data

if os.path.exists('brain_training_data.npy'):
  Brain_train_data = np.load('brain_training_data.npy', allow_pickle=True)
else:
  Brain_train_data = create_Brain_train_data()

Brain_X_train = np.array([i[0] for i in Brain_train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Brain_y_train = [i[1] for i in Brain_train_data]
Brain_Y_train_OHE = [[int(j == i) for j in range(max(Brain_y_train) + 1)] for i in Brain_y_train]
Brain_Y_train_reshaped = np.array(Brain_Y_train_OHE)

Brain_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, rescale = 1.0/255.)
Brain_datagen.fit(Brain_X_train)

def create_Breast_train_data():
  training_data = []
  for i in range(len(Breast_Train)):
    path = os.path.join(ROOT, Breast_Train[i])
    for img in os.listdir(path):
      if img.find('_mask') == -1:
        Full_path = os.path.join(path, img)
        img_data = cv2.imread(Full_path)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        if 'normal' in Full_path:
          training_data.append([img_data, 0])
        elif 'benign' in Full_path:
          training_data.append([img_data, 1])
        else:
          training_data.append([img_data, 2])
  shuffle(training_data)
  np.save('breast_training_data.npy', training_data)
  return training_data

if os.path.exists('breast_training_data.npy'):
  Breast_train_data = np.load('breast_training_data.npy', allow_pickle=True)
else:
  Breast_train_data = create_Breast_train_data()

Breast_X_train = np.array([i[0] for i in Breast_train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Breast_y_train = [i[1] for i in Breast_train_data]
Breast_Y_train_OHE = [[int(j == i) for j in range(max(Breast_y_train) + 1)] for i in Breast_y_train]
Breast_Y_train_reshaped = np.array(Breast_Y_train_OHE)


Breast_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, rescale = 1.0/255.)
Breast_datagen.fit(Breast_X_train)

if os.path.exists('Brain_VGG19.h5'):
    Brain_VGG19 = load_model('Brain_VGG19.h5')
else:
  Brain_VGG19 = Sequential()
  Brain_VGG19.add(VGG19(input_shape = (IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet'))
  for each_layer in Brain_VGG19.layers:
    each_layer.trainable = False
  Brain_VGG19.add(Flatten())
  Brain_VGG19.add(Dense(2, activation='softmax'))
  Brain_VGG19.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), metrics = ['accuracy'])
  Brain_VGG19.summary()
  Brain_VGG19.fit(Brain_datagen.flow(Brain_X_train, Brain_Y_train_reshaped), epochs=2, verbose='auto')
  Brain_VGG19.save('Brain_VGG19.h5')

if os.path.exists('Breast_VGG19.h5'):
  Breast_VGG19 = load_model('Breast_VGG19.h5')
else:
  Breast_VGG19 = Sequential()
  Breast_VGG19.add(VGG19(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet'))
  for each_layer in Breast_VGG19.layers:
    each_layer.trainable = False
  Breast_VGG19.add(Flatten())
  Breast_VGG19.add(Dense(3, activation='softmax'))
  Breast_VGG19.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                       metrics=['accuracy'])
  Breast_VGG19.summary()
  Breast_VGG19.fit(Breast_datagen.flow(Breast_X_train, Breast_Y_train_reshaped), epochs=2, verbose='auto')
  Breast_VGG19.save('Breast_VGG19.h5')

index = 0
Paths = []
Predictions = []
actual_classes = []
test_set = []
for folder in range(len(Brain_Breast_Test)):
  path = os.path.join(ROOT, Brain_Breast_Test[folder])
  for img in os.listdir(path):
    if img.find('_mask') == -1:
      Full_path = os.path.join(path, img)
      Paths.append(Full_path)
      img_test = cv2.imread(Full_path)
      img_test = cv2.resize(img_test, (IMG_SIZE, IMG_SIZE))
      img_test = img_test / 255.
      test_set.append(img_test)
      if 'Brain scans' in Full_path:
        actual_classes.append(0)
      else:
        actual_classes.append(1)
BB_X_test = np.array([i for i in test_set]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
prediction = Brain_Breast_VGG19.predict(BB_X_test)
for output in range(prediction.shape[0]):
  for target in range(2):
    if prediction[output, target] == max(prediction[output]):
      Predictions.append(target)
Predictions_np = np.array(Predictions)
actual_classes_np = np.array(actual_classes)
print("Brain_Breast_VGG19 accuracy = {}".format(accuracy_score(actual_classes_np, Predictions_np)))
print("Brain_Breast_VGG19 precision = {}".format(precision_score(actual_classes_np, Predictions_np)))
print("Brain_Breast_VGG19 recall = {}".format(recall_score(actual_classes_np, Predictions_np)))
ConfusionMatrixDisplay.from_predictions(actual_classes_np, Predictions_np)
plt.show()

Paths_np = np.array(Paths)
Brain_test_paths = Paths_np[Predictions_np == 0]
Brain_test_set = []
Brain_actual_classes = []
Brain_Predictions = []
for Brain_path in Brain_test_paths:
  if Brain_path.find('_mask') == -1:
    Brain_img_tst = cv2.imread(Brain_path)
    Brain_img_tst = cv2.resize(Brain_img_tst, (IMG_SIZE, IMG_SIZE))
    Brain_img_tst = Brain_img_tst / 255.
    Brain_test_set.append(Brain_img_tst)
    if 'No tumor' in Brain_path:
      Brain_actual_classes.append(0)
    else:
      Brain_actual_classes.append(1)
Brain_test_set_np = np.array([i for i in Brain_test_set]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Brain_prediction = Brain_VGG19.predict(Brain_test_set_np)
for Brain_output in range(Brain_prediction.shape[0]):
  for Brain_target in range(2):
    if Brain_prediction[Brain_output, Brain_target] == max(Brain_prediction[Brain_output]):
      Brain_Predictions.append(Brain_target)
Brain_Predictions_np = np.array(Brain_Predictions)
Brain_actual_classes_np = np.array(Brain_actual_classes)
print("Brain_VGG19 accuracy = {}".format(accuracy_score(Brain_actual_classes_np, Brain_Predictions_np)))
print("Brain_VGG19 precision = {}".format(precision_score(Brain_actual_classes_np, Brain_Predictions_np)))
print("Brain_VGG19 recall = {}".format(recall_score(Brain_actual_classes_np, Brain_Predictions_np)))
ConfusionMatrixDisplay.from_predictions(Brain_actual_classes_np, Brain_Predictions_np)
plt.show()

Breast_test_paths = Paths_np[Predictions_np == 1]
Breast_test_set = []
Breast_actual_classes = []
Breast_Predictions = []
for Breast_path in Breast_test_paths:
  if Breast_path.find('_mask') == -1:
    Breast_img_tst = cv2.imread(Breast_path)
    Breast_img_tst = cv2.resize(Breast_img_tst, (IMG_SIZE, IMG_SIZE))
    Breast_img_tst = Breast_img_tst / 255.
    Breast_test_set.append(Breast_img_tst)
    if 'normal' in Breast_path:
      Breast_actual_classes.append(0)
    elif 'benign' in Breast_path:
      Breast_actual_classes.append(1)
    else:
      Breast_actual_classes.append(2)
Breast_test_set_np = np.array([i for i in Breast_test_set]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Breast_prediction = Breast_VGG19.predict(Breast_test_set_np)
for Breast_output in range(Breast_prediction.shape[0]):
  for Breast_target in range(3):
    if Breast_prediction[Breast_output, Breast_target] == max(Breast_prediction[Breast_output]):
      Breast_Predictions.append(Breast_target)
Breast_Predictions_np = np.array(Breast_Predictions)
Breast_actual_classes_np = np.array(Breast_actual_classes)
print("Breast_VGG19 accuracy = {}".format(accuracy_score(Breast_actual_classes_np, Breast_Predictions_np)))
print("Breast_VGG19 precision = {}".format(
  precision_score(Breast_actual_classes_np, Breast_Predictions_np, average='micro')))
print(
  "Breast_VGG19 recall = {}".format(recall_score(Breast_actual_classes_np, Breast_Predictions_np, average='micro')))
ConfusionMatrixDisplay.from_predictions(Breast_actual_classes_np, Breast_Predictions_np)
plt.show()


def create_Brain_Tumor_train_data():
  X_train = []
  y_train = []
  for i in range(len(Brain_Tumor_Train)):
    path = os.path.join(ROOT, Brain_Tumor_Train[i])
    for img in os.listdir(path):
      if path.find('_mask') == -1:
        Full_path = os.path.join(path, img)
        img_data = cv2.imread(Full_path, 0)
        img_data = cv2.resize(img_data, (224, 224))
        X_train.append(img_data)
      else:
        Full_path = os.path.join(path, img)
        img_data = cv2.imread(Full_path, 0)
        img_data = cv2.resize(img_data, (224, 224))
        y_train.append(img_data)
  np.save('brain_tumor.npy', X_train)
  np.save('brain_tumor_masks.npy', y_train)
  return X_train, y_train


if os.path.exists('brain_tumor.npy') and os.path.exists('brain_tumor_masks.npy'):
  Brain_Tumor_X_train = np.load('brain_tumor.npy', allow_pickle=True)
  Brain_Tumor_y_train = np.load('brain_tumor_masks.npy', allow_pickle=True)
else:
  Brain_Tumor_X_train, Brain_Tumor_y_train = create_Brain_Tumor_train_data()

Brain_Tumor_X_train = np.array([i for i in Brain_Tumor_X_train]).reshape(-1, 224, 224, 1)
Brain_Tumor_y_train = np.array([i for i in Brain_Tumor_y_train]).reshape(-1, 224, 224, 1)

# Brain_Tumor_X_test = np.reshape(Brain_test_set_np[np.logical_and(Brain_Predictions_np == 1, Brain_actual_classes_np == 1)], () )
Brain_Tumor_X_test_paths = Brain_test_paths[Brain_Predictions_np == 1]
Brain_Tumor_X_test = []
Brain_Tumor_y_test = []
for Brain_Tumor_X_test_path in range(Brain_Tumor_X_test_paths.shape[0]):
  if 'Tumor' in Brain_Tumor_X_test_paths[Brain_Tumor_X_test_path]:
    Brain_Tumor = cv2.imread(Brain_Tumor_X_test_paths[Brain_Tumor_X_test_path], 0)
    Brain_Tumor = cv2.resize(Brain_Tumor, (224, 224))
    Brain_Tumor_X_test.append(Brain_Tumor)
    Brain_Tumor_X_test_paths[Brain_Tumor_X_test_path] = Brain_Tumor_X_test_paths[Brain_Tumor_X_test_path].replace(
      '.jpg', '.png')
    Brain_Tumor_X_test_paths[Brain_Tumor_X_test_path] = Brain_Tumor_X_test_paths[Brain_Tumor_X_test_path].replace(
      'TEST', 'TEST_masks')
    Brain_Tumor_mask = cv2.imread(Brain_Tumor_X_test_paths[Brain_Tumor_X_test_path], 0)
    Brain_Tumor_mask = cv2.resize(Brain_Tumor_mask, (224, 224))
    Brain_Tumor_y_test.append(Brain_Tumor_mask)
Brain_Tumor_X_test_np = np.array([i for i in Brain_Tumor_X_test]).reshape(-1, 224, 224, 1)
Brain_Tumor_y_test_np = np.array([i for i in Brain_Tumor_y_test]).reshape(-1, 224, 224, 1)
#     # Brain_Tumor_y_pred_np = Brain_Tumor_Unet_VGG19.predict(Brain_Tumor_X_test_np)
#
inply = Input((224, 224, 1,))

conv1 = Conv2D(2 ** 6, (3, 3), activation='relu', padding='same')(inply)
conv1 = Conv2D(2 ** 6, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D((2, 2), strides=2, padding='same')(conv1)
drop1 = Dropout(0.2)(pool1)

conv2 = Conv2D(2 ** 7, (3, 3), activation='relu', padding='same')(drop1)
conv2 = Conv2D(2 ** 7, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D((2, 2), strides=2, padding='same')(conv2)
drop2 = Dropout(0.2)(pool2)

conv3 = Conv2D(2 ** 8, (3, 3), activation='relu', padding='same')(drop2)
conv3 = Conv2D(2 ** 8, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D((2, 2), strides=2, padding='same')(conv3)
drop3 = Dropout(0.2)(pool3)

conv4 = Conv2D(2 ** 9, (3, 3), activation='relu', padding='same')(drop3)
conv4 = Conv2D(2 ** 9, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D((2, 2), strides=2, padding='same')(conv4)
drop4 = Dropout(0.2)(pool4)

convm = Conv2D(2 ** 10, (3, 3), activation='relu', padding='same')(drop4)
convm = Conv2D(2 ** 10, (3, 3), activation='relu', padding='same')(convm)

tran5 = Conv2DTranspose(2 ** 9, (2, 2), strides=2, padding='valid', activation='relu')(convm)
conc5 = Concatenate()([tran5, conv4])
conv5 = Conv2D(2 ** 9, (3, 3), activation='relu', padding='same')(conc5)
conv5 = Conv2D(2 ** 9, (3, 3), activation='relu', padding='same')(conv5)
drop5 = Dropout(0.1)(conv5)

tran6 = Conv2DTranspose(2 ** 8, (2, 2), strides=2, padding='valid', activation='relu')(drop5)
conc6 = Concatenate()([tran6, conv3])
conv6 = Conv2D(2 ** 8, (3, 3), activation='relu', padding='same')(conc6)
conv6 = Conv2D(2 ** 8, (3, 3), activation='relu', padding='same')(conv6)
drop6 = Dropout(0.1)(conv6)

tran7 = Conv2DTranspose(2 ** 7, (2, 2), strides=2, padding='valid', activation='relu')(drop6)
conc7 = Concatenate()([tran7, conv2])
conv7 = Conv2D(2 ** 7, (3, 3), activation='relu', padding='same')(conc7)
conv7 = Conv2D(2 ** 7, (3, 3), activation='relu', padding='same')(conv7)
drop7 = Dropout(0.1)(conv7)

tran8 = Conv2DTranspose(2 ** 6, (2, 2), strides=2, padding='valid', activation='relu')(drop7)
conc8 = Concatenate()([tran8, conv1])
conv8 = Conv2D(2 ** 6, (3, 3), activation='relu', padding='same')(conc8)
conv8 = Conv2D(2 ** 6, (3, 3), activation='relu', padding='same')(conv8)
drop8 = Dropout(0.1)(conv8)
outly = Conv2D(2 ** 0, (1, 1), activation='relu', padding='same')(drop8)

Brain_Tumor_Unet = Model(inputs=inply, outputs=outly, name='U-net')




# if os.path.exists('Brain_Tumor_Unet.h5'):
#     Brain_Tumor_Unet = load_model('Brain_Tumor_Unet.h5')
# else:
#   Brain_Tumor_Unet.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.00005),metrics=['accuracy', MeanIoU(num_classes=1)])
#   Brain_Tumor_Unet.fit(Brain_Tumor_X_train, Brain_Tumor_y_train, epochs=100)
#   Brain_Tumor_Unet.save('Brain_Tumor_Unet.h5')

def create_Breast_Malignant_train_data():
  X_train = []
  y_train = []
  path = os.path.join(ROOT, Breast_Malignant_Train)
  for img in os.listdir(path):
    if img.find('_mask') == -1:
      Full_path = os.path.join(path, img)
      img_data = cv2.imread(Full_path, 0)
      img_data = cv2.resize(img_data, (224, 224))
      X_train.append(img_data)
    else:
      Full_path = os.path.join(path, img)
      img_data = cv2.imread(Full_path, 0)
      img_data = cv2.resize(img_data, (224, 224))
      y_train.append(img_data)
  np.save('breast_malignant.npy', X_train)
  np.save('breast_malignant_masks.npy', y_train)
  return X_train, y_train

if os.path.exists('breast_malignant.npy') and os.path.exists('breast_malignant_masks.npy'):
  Breast_Malignant_X_train = np.load('breast_malignant.npy', allow_pickle=True)
  Breast_Malignant_y_train = np.load('breast_malignant_masks.npy', allow_pickle=True)
else:
  Breast_Malignant_X_train, Breast_Malignant_y_train = create_Breast_Malignant_train_data()

Breast_Malignant_X_train = np.array([i for i in Breast_Malignant_X_train]).reshape(-1, 224, 224, 1)
Breast_Malignant_y_train = np.array([i for i in Breast_Malignant_y_train]).reshape(-1, 224, 224, 1)

# Breast_Malignant_Unet = Model(inputs = inply, outputs = outly, name = 'U-net')
# Breast_Malignant_Unet.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.00005),metrics=['acc'])
# if os.path.exists('Brain_Tumor_Unet.h5'):
#     Brain_Tumor_Unet = load_model('Brain_Tumor_Unet.h5')
# else:
#   Brain_Tumor_Unet.fit(Brain_Tumor_X_train, Brain_Tumor_y_train, epochs=100)
#   Brain_Tumor_Unet.save('Brain_Tumor_Unet.h5')
#
# Breast_Malignant_X_test_paths = Breast_test_paths[Breast_Predictions_np == 2]
# Breast_Malignant_X_test = []
# Breast_Malignant_y_test = []
# for Breast_Malignant_X_test_path in range(Breast_Malignant_X_test_paths.shape[0]):
#   if 'malignant' in Breast_Malignant_X_test_paths[Breast_Malignant_X_test_path]:
#     Breast_Malignant = cv2.imread(Breast_Malignant_X_test_paths[Breast_Malignant_X_test_path], 0)
#     Breast_Malignant = cv2.resize(Breast_Malignant, (224, 224))
#     Breast_Malignant_X_test.append(Breast_Malignant)
#     Breast_Malignant_X_test_paths[Breast_Malignant_X_test_path] = Breast_Malignant_X_test_paths[Breast_Malignant_X_test_path].replace(').png', ')_mask.png')
#     print(Breast_Malignant_X_test_paths[Breast_Malignant_X_test_path])
#     Breast_Malignant_mask = cv2.imread(Breast_Malignant_X_test_paths[Breast_Malignant_X_test_path], 0)
#     Breast_Malignant_mask = cv2.resize(Breast_Malignant_mask, (224, 224))
#     Breast_Malignant_y_test.append(Breast_Malignant_mask)
# Breast_Malignant_X_test_np = np.array([i for i in Breast_Malignant_X_test]).reshape(-1, 224, 224, 1)
# Breast_Malignant_y_test_np = np.array([i for i in Breast_Malignant_y_test]).reshape(-1, 224, 224, 1)

def create_Breast_Benign_train_data():
  X_train = []
  y_train = []
  path = os.path.join(ROOT, Breast_Benign_Train)
  for img in os.listdir(path):
    if img.find('_mask') == -1:
      Full_path = os.path.join(path, img)
      img_data = cv2.imread(Full_path, 0)
      img_data = cv2.resize(img_data, (224, 224))
      X_train.append(img_data)
    else:
      directory = os.listdir(path)
      nextIndex = directory.index(img) + 1
      if (nextIndex != 0 and nextIndex != len(directory)) and ('_mask_' in directory[nextIndex]):
        Full_path = os.path.join(path, img)
        Full_path2 = os.path.join(path, directory[nextIndex])
        img_data = cv2.imread(Full_path, 0)
        img_data2 = cv2.imread(Full_path2, 0)
        img_data = cv2.resize(img_data, (224, 224))
        img_data2 = cv2.resize(img_data2, (224, 224))
        Complete_img_data = np.maximum(img_data, img_data2)
        y_train.append(Complete_img_data)
      else:
        Full_path = os.path.join(path, img)
        img_data = cv2.imread(Full_path, 0)
        img_data = cv2.resize(img_data, (224, 224))
        y_train.append(img_data)
  np.save('breast_benign.npy', X_train)
  np.save('breast_benign_masks.npy', y_train)
  return X_train, y_train

if os.path.exists('breast_benign.npy') and os.path.exists('breast_benign_masks.npy'):
  Breast_Benign_X_train = np.load('breast_benign.npy', allow_pickle=True)
  Breast_Benign_y_train = np.load('breast_benign_masks.npy', allow_pickle=True)
else:
  Breast_Benign_X_train, Breast_Benign_y_train = create_Breast_Benign_train_data()

Breast_Benign_X_train = np.array([i for i in Breast_Benign_X_train]).reshape(-1, 224, 224, 1)
Breast_Benign_y_train = np.array([i for i in Breast_Benign_y_train]).reshape(-1, 224, 224, 1)

Breast_Benign_X_test_paths = Breast_test_paths[Breast_Predictions_np == 1]
Breast_Benign_X_test = []
Breast_Benign_y_test = []
for Breast_Benign_X_test_path in range(Breast_Benign_X_test_paths.shape[0]):
  if 'benign' in Breast_Benign_X_test_paths[Breast_Benign_X_test_path]:
    Breast_Benign = cv2.imread(Breast_Benign_X_test_paths[Breast_Benign_X_test_path], 0)
    Breast_Benign = cv2.resize(Breast_Benign, (224, 224))
    Breast_Benign_X_test.append(Breast_Benign)
    Breast_Benign_X_test_paths[Breast_Benign_X_test_path] = Breast_Benign_X_test_paths[Breast_Benign_X_test_path].replace(')', ')_mask')
    check = Breast_Benign_X_test_paths[Breast_Benign_X_test_path].replace(')_mask', ')_mask_1')
    if os.path.exists(check):
      Breast_Benign_mask = cv2.imread(Breast_Benign_X_test_paths[Breast_Benign_X_test_path], 0)
      Breast_Benign_mask2 = cv2.imread(check, 0)
      Breast_Benign_mask = cv2.resize(Breast_Benign_mask, (224, 224))
      Breast_Benign_mask2 = cv2.resize(Breast_Benign_mask2, (224, 224))
      Complete_Benign_mask = np.maximum(Breast_Benign_mask, Breast_Benign_mask2)
      Breast_Benign_y_test.append(Complete_Benign_mask)
    else:
      Breast_Benign_mask = cv2.imread(Breast_Benign_X_test_paths[Breast_Benign_X_test_path], 0)
      Breast_Benign_mask = cv2.resize(Breast_Benign_mask, (224, 224))
      Breast_Benign_y_test.append(Breast_Benign_mask)


Breast_Benign_X_test_np = np.array([i for i in Breast_Benign_X_test]).reshape(-1, 224, 224, 1)
Breast_Benign_y_test_np = np.array([i for i in Breast_Benign_y_test]).reshape(-1, 224, 224, 1)

