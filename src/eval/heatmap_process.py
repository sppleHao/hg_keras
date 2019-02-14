
# coding: utf-8

# In[1]:


from scipy.ndimage import gaussian_filter, maximum_filter
import numpy as np


# In[4]:


def post_process_heatmap(heatmap,kp_confidence=0.2):
    kp_list = list()
    for i in range(heatmap.shape[-1]):
        # ignore last channel, background channel
        _map = heatmap[:,:,i]
        _map = gaussian_filter(_map,sigma=0.5)
        _nmsPeaks = non_max_supression(_map)
        
        y,x = np.where(_nmsPeaks = _nmsPeaks.max())
        
        if len(x)>0 and len(y)>0:
            kp_list.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            kp_list.append((0,0,0))
    return kp_list


# In[3]:


def non_max_supression(plain,window_size=3,threshold=1e-6):
    under_th_indicis = plain <threshold
    plain[under_th_indicis] = 0
    return plain * (plain == maximum_filter(plain,footprint=np.ones((window_size,window_size))))

