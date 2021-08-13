import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

!unzip /content/drive/MyDrive/New_covid_dataset.zip

IMAGE_SIZE = [224, 224]

train_path = "/content/New_covid_dataset/train"
test_path = "/content/New_covid_dataset/val"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set  = train_datagen.flow_from_directory(test_path,
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

xception =Xception(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in xception.layers:
    layer.trainable = False

base_model = Flatten()(xception.output)
prediction = Dense(2, activation='sigmoid')(base_model)
model = Model(inputs=xception.input, outputs=prediction)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

pd.DataFrame(r.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

model.save("/content/drive/MyDrive/Covid_19_model")




