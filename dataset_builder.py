#!/usr/bin/env python
# coding: utf-8

# In[13]:


import gdal
import os
import csv
import pandas as pd
import math
import numpy as np
import collections


# In[2]:


path = 'D://datasets//major//5552743//' #master folder path
csv_path='D://datasets//major//5552743//changed_excel_file.csv' #path of csv file

folder1 = 'IndianVillagesDataset_30m//imagery_res30_48bands//'
folder2 = 'IndianVillagesDataset_30m//masks_res30//'

dirs1 = os.listdir(path+folder1) #images from folder1
dirs2 = os.listdir(path+folder2) #images from folder2


# In[3]:


dirs1[:6]


# In[4]:


dataset = {}
csv_dict = {}

low_limit = 12
upper_limit = 23


# In[52]:



def calc_ndvi(image, mask, low_limit, upper_limit, size):
    band_percs = []
    
    for i in range(low_limit,upper_limit + 1):

        temp = image[i-1]*mask
        temp1 = np.where(np.logical_and(temp>0.2,temp<=0.6))
        perc = len(temp1[0])/size
        band_percs.append(perc)

    return sum(band_percs)/len(band_percs)


# In[65]:


def calc_evi_log(image, mask, low_limit, upper_limit, size):
    band_percs = []
    
    for i in range(low_limit,upper_limit + 1):
        if str(image[i-1][0][0]) != "nan":
            temp = image[i-1]*mask

            sumb = np.sum(temp)




            band_percs.append(math.log(sumb))

    return sum(band_percs)/len(band_percs)


# In[63]:


def calc_evi_mean(image, mask, low_limit, upper_limit, size):
    band_percs = []
    
    for i in range(low_limit,upper_limit + 1):
        if str(image[i-1][0][0]) != "nan":

            temp = image[i-1]*mask

            mean = np.mean(temp)


            band_percs.append(mean)

    return sum(band_percs)/len(band_percs)


# In[64]:


def calc_evi_median(image, mask, low_limit, upper_limit, size):
    band_percs = []
    
    for i in range(low_limit,upper_limit + 1):
        if str(image[i-1][0][0]) != "nan":
            temp = image[i-1]*mask

            median = np.median(temp)


            band_percs.append(median)

    return sum(band_percs)/len(band_percs)


# In[21]:


def infra_density(num_o_houses, mask):
    area = (np.count_nonzero(mask))*10
    den = num_o_houses/area
    return den


# In[ ]:





# In[11]:


csv = pd.read_csv(csv_path)
csv.head(10)
x = csv.sort_values(by=['Village Name','Census 2011 ID'])
x.head(10)


# In[12]:


status = x.drop([23528],axis=0)
status.head(10)
status['Electrified'][0]


# In[74]:


index = 0

village_name = list()
ndvi =list()
evi_log = list()
electrified = list()
evi_mean = list()
evi_median = list()
infra_den = list()

for file in dirs1[index:10000]:
    raster1 = gdal.Open(path+folder1+file)
    arr1 = raster1.ReadAsArray()

    raster2 = gdal.Open(path+folder2+file)
    arr2 = raster2.ReadAsArray()
    arr2 = arr2/255
    
    h = arr2.shape[0]
    w = arr2.shape[1]
    size = h*w
    
    village_name.append(file)
    ndvi.append(calc_ndvi(arr1, arr2, 12, 23, size))
    evi_log.append(calc_evi_log(arr1, arr2, 24, 35, size))
    evi_mean.append(calc_evi_mean(arr1, arr2, 24, 35, size))
    evi_median.append(calc_evi_median(arr1, arr2, 24, 35, size))
    electrified.append(status['Electrified'][index])
    infra_den.append(infra_density(abs(status['Number of Households'][index]),arr2))
    
    index+=1
print("donee")
    
dataset["village_name"] = village_name

dataset["ndvi"] = ndvi

dataset["evi_log"] = evi_log

dataset["evi_mean"] = evi_mean

dataset["evi_median"] = evi_median

dataset["infra_density"] = infra_den

dataset["electrified"] = electrified
    
    

# dataset is the final dict that contains ndvi info of all ndvi bands as well as electrification status

    


# In[73]:


type(dirs1)


# In[75]:


tt = pd.DataFrame.from_dict(dataset)


# In[ ]:





# In[76]:


tt.shape


# In[77]:


tt = tt[['village_name', 'ndvi', 'evi_log', 'evi_mean', 'evi_median',
       'infra_density', 'electrified']]


# In[78]:


tt.head(20)


# In[80]:


tt.to_csv(path+"database.csv", index=False)


# In[ ]:




