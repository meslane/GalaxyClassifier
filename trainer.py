import numpy
import os
import sys
import tensorflow as tf

#training parameters
batch_size = 32
height = 212
width = 212
seed = 314159
split = 0.2
epoch_size = 5
#GPUmemory = 4096 

#system calls to make the GPU work
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin") 
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

'''
gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.set_logical_device_configuration(gpu,
    [tf.config.LogicalDeviceConfiguration(memory_limit = GPUmemory)])
'''

print("TensorFlow version: {}".format(tf.__version__))

try:
    dir = str(sys.argv[1])
except (IndexError, FileNotFoundError):
    print("Error: cannot find given training data path")
    sys.exit(1)

#load training data
train = tf.keras.preprocessing.image_dataset_from_directory(
    dir,
    validation_split = split,
    subset = "training",
    seed = seed,
    image_size = (height, width),
    batch_size = batch_size,
    shuffle = True
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    dir,
    validation_split = split,
    subset = "validation",
    seed = seed,
    image_size = (height, width),
    batch_size = batch_size,
    shuffle = True
)   

names = train.class_names

print("Training classes: {}".format(names))

#cache training data
train = train.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
val = val.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

#create model
class_size = len(names)

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape = (height, width, 3)),
    tf.keras.layers.Conv2D(8, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(class_size)
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.summary()

#train model
history = model.fit(
    train,
    batch_size = batch_size,
    validation_data = val,
    epochs = epoch_size
)

#print results
'''
print(history.history['accuracy'])
print(history.history['val_accuracy'])
print(history.history['loss'])
print(history.history['val_loss'])
'''

#save model
try:
    model.save(str(sys.argv[2]))
except IndexError:
    print("Error: no path to save model given, model will not be saved")