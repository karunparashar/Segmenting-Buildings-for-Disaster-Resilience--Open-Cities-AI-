import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import skimage
from skimage.io import imread,imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split

# users need to just select dimension and channel
dimension = 256
channel = 3

DATA_DIR = '../data/'
REL_IMAGE_PATH = os.path.join(DATA_DIR, "images-"+str(dimension))
REL_MASK_PATH = os.path.join(DATA_DIR, "masks-"+str(dimension))

IMAGE_TRAIN_PATH  = os.path.join(REL_IMAGE_PATH, "train")
IMAGE_TEST_PATH  = os.path.join(REL_IMAGE_PATH, "test")

MASK_TRAIN_PATH = os.path.join(REL_MASK_PATH, "train")
MASK_TEST_PATH = os.path.join(REL_MASK_PATH, "test")

train_image_names = os.listdir(IMAGE_TRAIN_PATH)
test_image_names = os.listdir(IMAGE_TEST_PATH)

train_mask_names = os.listdir(MASK_TRAIN_PATH)
test_mask_names = os.listdir(MASK_TEST_PATH)

width = dimension
height = dimension
# #to create X and Y arrays to be later filled with images of both images and masks
x_train = np.zeros((len(train_image_names),height,width,channel),dtype = np.uint8)
y_train = np.zeros((len(train_mask_names),height,width,1),dtype = np.bool)

print("Reading train data......")

if os.path.isdir(IMAGE_TRAIN_PATH):
    i = 0
    for image_name in train_image_names:
        image_path = os.path.join(IMAGE_TRAIN_PATH, image_name)
        img = imread(image_path)[:,:,:channel]
        x_train[i] = resize(img,(height,width),mode ='edge',preserve_range =True)
        
        mask_path = os.path.join(MASK_TRAIN_PATH, image_name[:-4]+"_mask.png")
        mask = imread(mask_path)
        mask_resized = resize(mask, (height,width), mode ='edge', preserve_range = True)
        mask = np.expand_dims(mask_resized, axis=-1)
        y_train[i] = mask
        
        i += 1

    print("Train data read successfully!")
else:
    print("There is no folder: "+img_output_loc+". Please build the data-set using BuildNewDataSet notebook.")

print("Building model..")
#creating U-Net architecture
inputs = tf.keras.layers.Input((width,height,channel))
s = tf.keras.layers.Lambda(lambda x : x/255)(inputs)
c1 = tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)


c5 = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c5)


u6 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides = (2,2),padding = 'same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(u6)
c6 = tf.keras.layers.Dropout(0.3)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides = (2,2),padding = 'same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides = (2,2),padding = 'same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16,(2,2),strides = (2,2),padding = 'same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1],axis = 3)
c9 = tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',kernel_initializer='he_normal',padding = 'same')(c9)

outputs = tf.keras.layers.Conv2D(1,(1,1),activation = "sigmoid")(c9)

model = tf.keras.Model(inputs = [inputs],outputs = [outputs])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())

model_path="v3_binary_crossentropy_256x256-{epoch:02d}-{val_acc:.2f}.hdf5"
clbk = [tf.keras.callbacks.ModelCheckpoint(model_path, verbose= 1,save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience = 3, monitor = 'val_acc', mode = 'max'),
        tf.keras.callbacks.TensorBoard(log_dir = "logs")]

print("Training model....")

results = model.fit(x_train,y_train, epochs=20,
                   verbose = 1,use_multiprocessing=True,
                   callbacks=clbk, validation_split=0.2)

print("Model trained....")




