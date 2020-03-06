from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import models
from keras.applications.vgg16 import preprocess_input, decode_predictions


model = load_model('cats_and_dogs_small_2.h5')
#model.summary()

model2=load_model('cats_and_dogs_small_3.h5')
#model2.summary()

#conv_base = VGG16(weights='imagenet', include_top=False)

sample_img=os.getcwd()+'/dogs-vs-cats/train/dog.120.jpg'

img = image.load_img(sample_img, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

prediction = model2.predict(img_tensor)
print(prediction)
plt.imshow(img_tensor[0])
plt.show()

##activation map
#Extracts the outputs of the top eight layers
layer_outputs = [layer.get_output_at(-1) for layer in model2.layers[:8]]
#Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model2.input, outputs=layer_outputs)

#Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)

#the layer contains 32 channels
first_layer_activation = activations[0]

#plot the 17th channel
plt.matshow(first_layer_activation[0, :, :, 1], cmap='viridis')
plt.show()
