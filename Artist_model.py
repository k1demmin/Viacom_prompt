import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow as tf

train_dir = os.path.join('/Users/malachi/Documents/Programming/Viacom2/CV-home-data/')
test_dir = os.path.join('/Users/malachi/Documents/Programming/Viacom2/CV-home-data/')

def image_gen_w_aug(train_parent_directory, test_parent_directory):
    train_datagen = ImageDataGenerator(rescale=1/225,
                                      rotation_range = 30,
                                      zoom_range = 0.2,
                                      width_shift_range = 0.1,
                                      height_shift_range=0.1,
                                      validation_split = 0.15)
    
    test_datagen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                       target_size = (75,75),
                                                       batch_size = 6,
                                                       class_mode = 'categorical',
                                                       subset='training')
    val_generator = train_datagen.flow_from_directory(train_parent_directory,
                                                     target_size=(75,75),
                                                     batch_size=6,
                                                     class_mode = 'categorical',
                                                     subset = 'validation')
    
    test_generator = test_datagen.flow_from_directory(test_parent_directory,
                                                     target_size=(75,75),
                                                     batch_size = 6,
                                                     class_mode='categorical')
    return train_generator, val_generator, test_generator

train_generator, validation_generator, test_generator = image_gen_w_aug(train_dir,test_dir)

def model_output_for_TL (pre_trained_model, last_output):
    x = Flatten()(last_output)
    
    #Dense Hidden Layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    #Output Neuron
    x = Dense(6, activation='softmax')(x)
    
    model = tf.keras.Model(pre_trained_model.input, x)
    
    return model

pre_trained_model = InceptionV3(input_shape = (75,75,3),
                               include_top = False,
                               weights = 'imagenet')

for layer in pre_trained_model.layers:
    layer.trainable=False

last_layer = pre_trained_model.get_layer('mixed5')
last_output = last_layer.output
model_TL = model_output_for_TL(pre_trained_model,last_output)

model_TL.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

history_TL = model_TL.fit(train_generator,
                         steps_per_epoch=2,
                         epochs = 24,
                         verbose=1,
                         validation_data = validation_generator)

tf.keras.models.save_model(model_TL,'my_model.hdf5')
