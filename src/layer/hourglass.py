
# coding: utf-8

# In[2]:


import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../eval/")

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
import datetime
import scipy.misc
from data_process import normalize
import numpy as np
import os

from layers import create_hourglass_network,residual
from mpii_datagen import MPIIDataGen
from eval_callback import EvalCallBack


# In[5]:


class HourglassNet(object):
    def __init__(self,num_classes,num_stacks,num_channels,inres,outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.num_channels = num_channels
        self.inres = inres
        self.outres = outres
    
    def build_model(self,mobile=False,show=False):
        if mobile:
            pass
        else:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, residual)
        #show model layers
        if show:
            self.model.summary()
    def train(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen('../../data/mpii/mpii_annotations.json', '../../data/mpii/images',inres=self.inres, outres=self.outres, is_train=True)
        
        train_gen = train_dataset.generator(batch_size=batch_size,num_stack=self.num_stacks,sigma=1,is_shuffle=True,rot_flag=True,scale_flag=True,flip_flag=True)
        
        csvlogger = CSVLogger(os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
        
        checkpoint = EvalCallBack(model_path, self.inres, self.outres)
        
        xcallbacks = [csvlogger, checkpoint]
        
        self.model.fit_generator(generator=train_gen,steps_per_epoch =train_dataset.get_dataset_size()// batch_size,epochs=epochs, callbacks=xcallbacks)
        
    def resume_train(self,batch_size,exist_model_path,init_epoch,epochs):
        self.load_model(exist_model_path)
        
        self.model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])
        
        train_dataset = MPIIDataGen('../../data/mpii/mpii_annotations.json', '../../data/mpii/images',inres=self.inres, outres=self.outres, is_train=True)
        
        train_gen = train_dataset.generator(batch_size=batch_size,num_stack=self.num_stacks,sigma=1,is_shuffle=True,rot_flag=True,scale_flag=True,flip_flag=True)
        
        model_dir = os.path.dirname(os.path.abspath(exist_model_path))
        
        csvlogger = CSVLogger(os.path.join(model_path,'csv_train_'+str(datetime.datetime.now().strftime('%H:%M'))+'.csv'))
        
        checkpoint = EvalCallBack(model_dir, self.inres, self.outres)

        xcallbacks = [csvlogger, checkpoint]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // batch_size,initial_epoch=init_epoch, epochs=epochs, callbacks=xcallbacks)
        
        
    def load_model(self , model_path):
        self.model = load_model(model_path)
    
    def inference_rgb(self, rgbdata, orgshape, mean=None):
        scale = (orgshape[0] * 1.0 / self.inres[0], orgshape[1] * 1.0 / self.inres[1])
        imgdata = scipy.misc.imresize(rgbdata, self.inres)

        if mean is None:
            mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)

        imgdata = normalize(imgdata, mean)

        input = imgdata[np.newaxis, :, :, :]

        out = self.model.predict(input)
        return out[-1], scale

    def inference_file(self, imgfile, mean=None):
        imgdata = scipy.misc.imread(imgfile)
        ret = self.inference_rgb(imgdata, imgdata.shape, mean)
        return ret

