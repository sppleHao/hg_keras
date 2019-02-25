
# coding: utf-8

# In[2]:


from keras.models import *
from keras.layers import *
from keras.optimizers import Adam , RMSprop
from keras.losses import mean_squared_error
import keras.backend as K
import tensorflow as tf


# In[74]:


def convBlock(inputs,num_out_channels,name):
    x = Conv2D(num_out_channels // 2,kernel_size=(1,1),activation='relu',padding='same',name=name+'_conv_1x1_x1')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(num_out_channels // 2,kernel_size=(3,3),activation='relu',padding='same',name=name+'_conv_3x3_x2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(num_out_channels , kernel_size=(1,1),activation='relu',padding='same',name=name+'_conv_1x1_x3')(x)
    x = BatchNormalization()(x)
    return x

def convBlock_mobile(inputs,num_out_channels,name):
    x = SeparableConv2D(num_out_channels // 2,kernel_size=(1,1),activation='relu',padding='same',name=name+'_conv_1x1_x1')(inputs)
    x = BatchNormalization()(x)
    x = SeparableConv2D(num_out_channels // 2,kernel_size=(3,3),activation='relu',padding='same',name=name+'_conv_3x3_x2')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(num_out_channels , kernel_size=(1,1),activation='relu',padding='same',name=name+'_conv_1x1_x3')(x)
    x = BatchNormalization()(x)
    return x

# In[63]:


def skipLayer(inputs,num_out_channels,name):
    if K.int_shape(inputs)[-1]==num_out_channels:
        skip=inputs
    else:
        skip=Conv2D(num_out_channels,kernel_size=(1,1),activation='relu',padding='same',name=name+'_skip_conv')(inputs)
    return skip

def skipLayer_mobile(inputs,num_out_channels,name):
    if K.int_shape(inputs)[-1]==num_out_channels:
        skip=inputs
    else:
        skip=SeparableConv2D(num_out_channels,kernel_size=(1,1),activation='relu',padding='same',name=name+'_skip_conv')(inputs)
    return skip


# In[68]:


def residual(inputs, num_out_channels, name):
    convb = convBlock(inputs,num_out_channels,name+'convBlock')
    skip = skipLayer(inputs,num_out_channels,name+'skipLayer')
    add = Add()([convb,skip])  
    return add

def residual_mobile(inputs, num_out_channels, name):
    convb = convBlock_mobile(inputs,num_out_channels,name+'convBlock')
    skip = skipLayer_mobile(inputs,num_out_channels,name+'skipLayer')
    add = Add()([convb,skip])  
    return add

# In[65]:


def create_hourglass_network(num_classes, num_stacks, num_channels, inres, outres, residual_type):
    inputs = Input(shape=(inres[0],inres[1],3))
    
    front_features = create_front_module(inputs,num_channels,residual_type)
    
    head_next_stage = front_features
    
    outputs = []
    for i in range(num_stacks):
        head_next_stage,head_to_loss = hourglass_module(head_next_stage,num_classes,num_channels,residual_type,i)
        outputs.append(head_to_loss)
        
    model = Model(inputs=inputs,outputs=outputs)
    rms =RMSprop(lr=5e-4)
    model.compile(optimizer=rms,loss=mean_squared_error,metrics=["accuracy"])
    
    return model


# In[21]:


def create_front_module(inputs,num_channels,residual_type):
    x = Conv2D(64,kernel_size=(7,7),strides=(2,2),activation='relu',padding='same',name='front_conv_7x7_x1')(inputs)
    x = BatchNormalization()(x)
    
    x = residual_type(x,num_channels//2,'front_residual_x1')
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    
    x = residual_type(x,num_channels//2,'front_residual_x2')
    x = residual_type(x,num_channels,'front_residual_x3')
    
    return x


# In[55]:


def hourglass_module(inputs, num_classes, num_channels, residual_type, hg_id):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(inputs,residual_type,num_channels,hg_id)
    
    # create right features, connect with left features
    rf1 =create_right_half_blocks(left_features,residual_type,num_channels,hg_id)
    
    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    head_next_stage,head_parts = create_heads(inputs,rf1,num_classes,num_channels,hg_id)
    
    return head_next_stage,head_parts


# In[26]:


def create_left_half_blocks(inputs, residual_type, num_channels,hg_id):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution
    hg_name = 'hg_'+str(hg_id)
    
    f1 = residual_type(inputs,num_channels,hg_name+'_l1')
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(f1)
    
    f2 = residual_type(x,num_channels,hg_name+'_l2')
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(f2)
    
    f4 = residual_type(x,num_channels,hg_name+'_l4')
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(f4)
    
    f8 = residual_type(x,num_channels,hg_name+'_l8')
    
    return (f1,f2,f4,f8)


# In[54]:


def create_right_half_blocks(left_features,residual_type,num_channels,hg_id):
    
    name = 'hg_'+ str(hg_id)
    
    lf1 ,lf2 ,lf4 ,lf8 = left_features
    
    rf8 = bottom_layer(lf8, residual_type, num_channels,hg_id)
    
    rf4 = connect_left_to_right(lf4,rf8,residual_type,num_channels,name+'_rf4')
    
    rf2 = connect_left_to_right(lf2,rf4,residual_type,num_channels,name+'_rf2')
    
    rf1 = connect_left_to_right(lf1,rf2,residual_type,num_channels,name+'_rf1')

    return rf1


# In[52]:


def bottom_layer(lf8, residual_type, num_channels,hg_id):
    # blocks in lowest resolution
    # 3 residual blocks + Add
    
    name = 'hg_'+ str(hg_id)
    
    lf8_connect = residual_type(lf8,num_channels,name+'_lf8')
    
    x = residual_type(lf8,num_channels,name+'lf8_resi_x1_') #error reason: tensor name must start with letter
    x = residual_type(x,num_channels,name+'lf8_resi_x2_')
    x = residual_type(x,num_channels,name+'lf8_resi_x3_')
    
    add = Add()([x,lf8_connect])
    
    return add


# In[46]:


def connect_left_to_right(lf,rf,residual_type,num_channels,name):
    
    # left with 1 residual
    # right upsampling
    # connect layers and with 1 residual 
    
    xleft = residual_type(lf,num_channels,name+'_connect_left_resi_')
    xright = UpSampling2D()(rf)
    add = Add()([xleft,xright])
    out = residual_type(add,num_channels,name+'_connect_')
    
    return out


# In[70]:


def create_heads(pre_layer_features, rf1, num_classes, num_channels, hg_id):
    # two head, one head to next stage, one head to intermediate features
    
    name = 'head_'+str(hg_id)
    
    head = Conv2D(num_channels,kernel_size=(1,1),activation='relu',padding='same',name=name+'_conv_1x1_x1')(rf1)
    head = BatchNormalization()(head)
    
    # for head as intermediate supervision, use 'linear' as activation.
    head_parts = Conv2D(num_classes,kernel_size=(1,1),activation='linear',padding='same',name=name + '_conv_1x1_parts')(head)
    
    # use linear activations
    head_connect = Conv2D(num_channels,kernel_size=(1,1),activation='linear',padding='same',name=name +'_conv_1x1_x2')(head)
    head_parts_connect = Conv2D(num_channels,kernel_size=(1,1),activation='linear',padding='same',name=name+'_conv_1x1_x3')(head_parts)
    
    # connect 
    head_next_stage = Add()([head_connect,head_parts_connect,pre_layer_features])
    
    return head_next_stage,head_parts

def euclidean_loss(x, y):
    return K.sqrt(K.sum(K.square(x - y)))