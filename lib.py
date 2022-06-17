import rasterio
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image

from time import time
import csv
import os
import numpy as np

import matplotlib.pyplot as plt
from random import shuffle
from itertools import compress


min_tlenght_cut=2


class InMemoryDataset(torch.utils.data.Dataset):
  
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


def processAndAugment(data):
    (x,y) = data
    im,label = x.copy(), y.copy()

    # convert to PIL for easier transforms
    im1 = Image.fromarray(im[0])
    im2 = Image.fromarray(im[1])
    label = Image.fromarray(label.squeeze())

  # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im1, (256, 256))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        label = F.vflip(label)

    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
    im = torch.stack([transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()])
    im = norm(im)
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(.003) * label.lt(.004)):
        label *= 255
    #label = label.round()

    return im, label

def processTestIm(data):
    (x,y) = data
    im,label = x.copy(), y.copy()
    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

    # convert to PIL for easier transforms
    im_c1 = Image.fromarray(im[0]).resize((512,512))
    im_c2 = Image.fromarray(im[1]).resize((512,512))
    label = Image.fromarray(label.squeeze()).resize((512,512))

    im_c1s = [F.crop(im_c1, 0, 0, 256, 256), F.crop(im_c1, 0, 256, 256, 256),
            F.crop(im_c1, 256, 0, 256, 256), F.crop(im_c1, 256, 256, 256, 256)]
    im_c2s = [F.crop(im_c2, 0, 0, 256, 256), F.crop(im_c2, 0, 256, 256, 256),
            F.crop(im_c2, 256, 0, 256, 256), F.crop(im_c2, 256, 256, 256, 256)]
    labels = [F.crop(label, 0, 0, 256, 256), F.crop(label, 0, 256, 256, 256),
            F.crop(label, 256, 0, 256, 256), F.crop(label, 256, 256, 256, 256)]

    ims = [torch.stack((transforms.ToTensor()(x).squeeze(),
                    transforms.ToTensor()(y).squeeze()))
                    for (x,y) in zip(im_c1s, im_c2s)]

    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)
  
    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    #labels = labels.round()
  
    return ims, labels

def getArrFlood(fname):
    return rasterio.open(fname).read()

def download_flood_water_data_from_list(l):
    i = 0
    tot_nan = 0
    tot_good = 0
    flood_data = []
    for (im_fname, mask_fname) in l:
        if not os.path.exists(os.path.join("files/", im_fname)):
            continue
        arr_x = np.nan_to_num(getArrFlood(os.path.join("files/", im_fname)))
        arr_y = getArrFlood(os.path.join("files/", mask_fname))
        arr_y[arr_y == -1] = 255 

        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51

        if i % 100 == 0:
              print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x,arr_y))

    return flood_data

def load_flood_train_data(input_root, label_root):
    fname = "/nfs/kloe/einstein4/martino/WQeMS/flood_train_data.csv"
    training_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            training_files.append(tuple((input_root+line[0], label_root+line[1])))

    return download_flood_water_data_from_list(training_files)

def load_flood_valid_data(input_root, label_root):
    fname = "/nfs/kloe/einstein4/martino/WQeMS/flood_valid_data.csv"
    validation_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            validation_files.append(tuple((input_root+line[0], label_root+line[1])))

    return download_flood_water_data_from_list(validation_files)

def load_flood_test_data(input_root, label_root):
    fname = "/nfs/kloe/einstein4/martino/WQeMS/flood_test_data.csv"
    testing_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            testing_files.append(tuple((input_root+line[0], label_root+line[1])))

    return download_flood_water_data_from_list(testing_files)


def f(y):
    a=0
    b=0
    cons_idx_list=list()
    for i in 1+np.arange(len(y)-1):
        if (y[i]-y[i-1]==0 or y[i]-y[i-1]==1):
            #a=cons_b_idx[i-1]
            b=i
            if i==len(y)-1:
                x=np.intersect1d(np.arange(a,b+1),np.where(y!=255)[0])
                if len(x)>0:
                    cons_idx_list.append(x)
        else:
            x=np.intersect1d(np.arange(a,b+1),np.where(y!=255)[0])
            if len(x)>0:
                cons_idx_list.append(x)
            a=i
        #print(i)
    return cons_idx_list

    #print(i,a,b)
    
def get_timeseries_train(dataset):
    y_list=list()
    hoty_list=list()
    #ch_0_list=list()
    #ch_1_list=list()
    data=list()
    counter=0
    
    t_size=len(dataset[:])
    for i in np.arange(256):
        for j in np.arange(256):
            #if counter==10000:
            #    break
            #j=100
            ch_0=list()
            ch_1=list()
            y=list()
            for t in np.arange(t_size):#len(train_data)):
                ch_0.append(float(dataset[t][0][0][i,j]))
                ch_1.append(float(dataset[t][0][1][i,j]))
                y.append(int(dataset[t][1][i,j]))

            ch_0=np.array(ch_0)
            ch_1=np.array(ch_1)    
            y=np.array(y)


            idx=f(y)
            for k in np.arange(len(idx)):
                y_list.append(np.array(y[idx[k]]))
                if 1 in y[idx[k]]:
                    hoty_list.append([1])
                else:
                    hoty_list.append([0])
                #ch_0_list.append(np.array(ch_0[idx[k]]))
                #ch_1_list.append(np.array(ch_1[idx[k]]))
                data.append(np.array([np.array(ch_0[idx[k]]),np.array(ch_1[idx[k]])]).reshape(2,len(idx[k])))
            counter+=1
            if (counter)%1000==0:
                print(counter/256**2,len(y_list))
        #print(len(y_list))
    return data,y_list

def get_timeseries_test(dataset):
    y_list=list()
    hoty_list=list()
    #ch_0_list=list()
    #ch_1_list=list()
    data=list()
    counter=0
    
    t_size=len(dataset[:])
    for i in np.arange(256):
        for j in np.arange(256):
            #if counter==10000:
            #    break
            ch_0=list()
            ch_1=list()
            y=list()
            for t in np.arange(t_size):#len(train_data)):
                ch_0.append(float(dataset[t][0][0][0][i,j]))
                ch_1.append(float(dataset[t][0][0][1][i,j]))
                y.append(int(dataset[t][1][0][i,j]))

            ch_0=np.array(ch_0)
            ch_1=np.array(ch_1)    
            y=np.array(y)


            idx=f(y)
            for k in np.arange(len(idx)):
                y_list.append(np.array(y[idx[k]]))
                if 1 in y[idx[k]]:
                    hoty_list.append([1])
                else:
                    hoty_list.append([0])
                #ch_0_list.append(np.array(ch_0[idx[k]]))
                #ch_1_list.append(np.array(ch_1[idx[k]]))
                data.append(np.array([np.array(ch_0[idx[k]]),np.array(ch_1[idx[k]])]).reshape(2,len(idx[k])))
            counter+=1
            if (counter)%1000==0:
                print(counter/256**2,len(y_list))
        #print(len(y_list))
    return data,y_list

def get_rawtimeseries(dataset):
    y_list=list()
    hoty_list=list()
    #ch_0_list=list()
    #ch_1_list=list()
    data=list()
    counter=0
    
    t_size=len(dataset[:])
    for i in np.arange(256):
        for j in np.arange(256):
            #if counter==10000:
            #    break
            
            ch_0=list()
            ch_1=list()
            y=list()
            for t in np.arange(t_size):#len(train_data)):
                ch_0.append(float(dataset[t][0][0][i,j]))
                ch_1.append(float(dataset[t][0][1][i,j]))
                y.append(int(dataset[t][1][0][i,j]))

            ch_0=np.array(ch_0)
            ch_1=np.array(ch_1)    
            y=np.array(y)


            idx=f(y)
            for k in np.arange(len(idx)):
                y_list.append(np.array(y[idx[k]]))
                if 1 in y[idx[k]]:
                    hoty_list.append([1])
                else:
                    hoty_list.append([0])
                #ch_0_list.append(np.array(ch_0[idx[k]]))
                #ch_1_list.append(np.array(ch_1[idx[k]]))
                data.append(np.array([np.array(ch_0[idx[k]]),np.array(ch_1[idx[k]])]).reshape(2,len(idx[k])))
            counter+=1
            if (counter)%1000==0:
                print(counter/256**2,len(y_list))
        #print(len(y_list))
    return data,y_list

def create_dataset(data,y_list):
    selected_signal_idx=list()
    selected_bkg_idx=list()
    for k in np.arange(len(y_list)):
        if y_list[k].sum()>min_tlenght_cut:
            selected_signal_idx.append(k)
        if y_list[k].sum()==0 and len(y_list[k])>min_tlenght_cut:
            selected_bkg_idx.append(k)
    selected_signal_idx=np.array(selected_signal_idx)
    selected_bkg_idx=np.array(selected_bkg_idx)
    
    selected_signal_mask=np.zeros(len(y_list),dtype=bool)
    selected_bkg_mask=np.zeros(len(y_list),dtype=bool)
    for k in selected_signal_idx:
            selected_signal_mask[k]=True
    for k in selected_bkg_idx:
            selected_bkg_mask[k]=True

    selected_data_sig=list(compress(data, selected_signal_mask))
    selected_data_bkg=list(compress(data, selected_bkg_mask))
    selected_y_sig=list(compress(y_list,selected_signal_mask))
    selected_y_bkg=list(compress(y_list,selected_bkg_mask))

    selected_data=selected_data_sig+selected_data_bkg
    #selected_data.append(selected_data_bkg)
    selected_y=np.ones(len(selected_data_sig))
    selected_y=np.append(selected_y,np.zeros(len(selected_data_bkg)))

    print('data size:',len(selected_data),len(selected_y))

    c = list(zip(selected_data, selected_y))

    random.shuffle(c)

    selected_data, selected_y = zip(*c)
    selected_y=np.array(selected_y)
    
    return selected_data, selected_y

def reorder_ragged_data(selected_data,selected_y):
    
    l=list([])
    size=len(selected_y)
    print(size)
    for k in np.arange(size):
        l.append(selected_data[k].shape[1])

    max_tlenght=max(l)
    ragged_data=list()
    ragged_y=list()
    recurrence=list()
    for s in np.arange(min_tlenght_cut+1,max_tlenght):
        count=0
        sel_data_tmp=list()
        sel_y_tmp=list()
        for k in np.arange(selected_y.shape[0]):
            if selected_data[k][0].shape[0]==s:
                #print(count)
                sel_data_tmp.append(selected_data[k].T)
                sel_y_tmp.append(selected_y[k])
                count+=1
        recurrence.append(count)
        sel_data_tmp=np.array(sel_data_tmp)
        sel_y_tmp=np.array(sel_y_tmp)

        ragged_data.append(sel_data_tmp)
        ragged_y.append(sel_y_tmp)
    return ragged_data,ragged_y,max_tlenght



def reorder_split_data(selected_data,selected_y):

    ragged_data,ragged_y,max_tlenght=reorder_ragged_data(selected_data,selected_y)
    
    sel_data_flatten_ch0_sig=list()
    sel_data_flatten_ch1_sig=list()
    sel_data_flatten_ch0_bkg=list()
    sel_data_flatten_ch1_bkg=list()
    for k in np.arange(max_tlenght-min_tlenght_cut-1):
        #print(k)
        sel_data_flatten_ch0_sig.append(ragged_data[k][ragged_y[k]==1,:,0].flatten())
        sel_data_flatten_ch1_sig.append(ragged_data[k][ragged_y[k]==1,:,1].flatten())
        sel_data_flatten_ch0_bkg.append(ragged_data[k][ragged_y[k]==0,:,0].flatten())
        sel_data_flatten_ch1_bkg.append(ragged_data[k][ragged_y[k]==0,:,1].flatten())

    sel_data_flatten_ch0_sig=np.concatenate(sel_data_flatten_ch0_sig).ravel()
    sel_data_flatten_ch1_sig=np.concatenate(sel_data_flatten_ch1_sig).ravel()
    sel_data_flatten_ch0_bkg=np.concatenate(sel_data_flatten_ch0_bkg).ravel()
    sel_data_flatten_ch1_bkg=np.concatenate(sel_data_flatten_ch1_bkg).ravel()
    return sel_data_flatten_ch0_sig,sel_data_flatten_ch1_sig,sel_data_flatten_ch0_bkg,sel_data_flatten_ch1_bkg


