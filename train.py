import keras,os
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import numpy as np
import h5py
from keras.applications.vgg16 import VGG16

def getVGGModel():
    input_tensor = Input(shape=(224,224,3))
    vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = vgg_model.get_layer('block5_pool').output

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(5, activation='softmax')(x)
    cus_model = Model(input=vgg_model.input, output=x)
    return cus_model

model = getVGGModel()

for layer in model.layers[:19]:
    layer.trainable = False

opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print(model.summary())

datagen = ImageDataGenerator()

train_it = datagen.flow_from_directory('Dataset/Train/', target_size=(224,224), class_mode='categorical', batch_size=16, shuffle=True)
val_it = datagen.flow_from_directory('Dataset/Validation/', target_size=(224,224), class_mode='categorical', batch_size=16, shuffle=True)

model.fit_generator(train_it, 
                    steps_per_epoch=325, 
                    validation_data=val_it, 
                    validation_steps=82, 
                    epochs=5)
