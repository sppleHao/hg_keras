
# coding: utf-8

# In[13]:


import sys
sys.path.insert(0, "../data_gen/")

import keras
import os
import datetime
from time import time
from mpii_datagen import MPIIDataGen
from eval_heatmap import cal_heatmap_acc


# In[12]:


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath, inres, outres):
        self.foldpath = foldpath
        self.inres = inres
        self.outres = outres
        
    def get_folder_path(self):
        return self.foldpath
    
    def run_eval(self,epoch):
        val_data = MPIIDataGen(jsonfile='../../data/mpii_annotations.json',imgpath='../../data/mpii/images',inres=self.inrese,outres=self.outres,is_train=False)
        
        total_success , total_fail = 0 ,0
        threshold = 0.5
        
        count = 0
        batch_size = 8
        
        for _img , _gthmap , _meta in val_data.generator(batch_size,8,sigma=2,is_shuffle=False,with_meta=True):
            
            count += batch_size
            if count > val_data.get_dataset_size():
                break
                
            out = self.model.predict(_img)
            
            suc,bad = cal_heatmap_acc(out[-1],_meta,threshold)
            
            total_success +=suc
            total_fail += bad
        
        acc = total_success * 1.0 / (total_success+total_fail)
        
        print('Eval Accuracy ',acc,' @ Epoch ',epoch)
        
        with open(os.path.join(self.get_folder_path(),'val.txt'),'a+') as xfile:
            xfile.write('Epoch '+ str(epoch) + ':' + str(acc) +'\n')
        
    def on_epoch_end(self,epoch,logs=None):
        #save model
        
        model_name = os.path.join(self.foldpath,'hg_epoch'+str(epoch)+'.h5')
        self.model.save(model_name)
        
        print('Saving model to ',model_name)
        
        self.run_eval(epoch)

