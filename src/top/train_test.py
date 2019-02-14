
# coding: utf-8

# In[1]:


import sys

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../layer/")

import argparse
import os
import tensorflow as tf
from keras import backend as k
from hourglass import HourglassNet


# In[4]:


batch_size = 8
model_path = '../../trained_models/'
resume = False
num_stacks = 2
epochs = 1
init_epoch = 0


# In[6]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    
# TensorFlow wizardry
config = tf.ConfigProto()
    
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 1.0
    
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

xnet = HourglassNet(num_classes=16, num_stacks=num_stacks, num_channels=256, inres=(256, 256),outres=(64, 64))

if resume:
    xnet.resume_train(batch_size=batch_size,
                      model_path=model_path,
                      init_epoch=init_epoch, epochs=args.epochs)
else:
    xnet.build_model(show=True)
    xnet.train(epochs=epochs, model_path=model_path, batch_size=batch_size)
    