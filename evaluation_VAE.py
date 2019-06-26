# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:20:55 2019

@author: Joris
"""

from __future__ import print_function
import argparse
import torch
import sys
import os
import math
import torch.utils.data
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Laplace, Independent, Bernoulli, Gamma, Uniform, Beta
from torch.distributions.kl import kl_divergence
from sklearn.manifold import TSNE
import pickle

import model_VAE
import data_loader
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
 

        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(torch.device("cpu"))
#    model.double()
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
#    print('train loss', checkpoint['train loss'])
#    print('test loss', checkpoint['test loss'])

    return model, checkpoint

def split_labels(z, y):
    label = {0 : [[],[]],
              1 : [[],[]],
              2 : [[],[]],
              3 : [[],[]],
              4 : [[],[]],
              5 : [[],[]],
              6 : [[],[]]} 
    for i in range(len(z)):
        label[y[i]][0].append(z[i])
        label[y[i]][1].append(y[i])
    return label

def cluster_eval(x, y, clusters, labels , k = 0, name = None):
    lesion_type_dict = {
        0: 'Actinic keratoses',
        1: 'Basal cell carcinoma',
        2: 'Benign keratosis-like lesions ',
        3: 'Dermatofibroma',
        4: 'dermatofibroma',
        5: 'Melanocytic nevi',
        6: 'Vascular lesions'
    }
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(x)
    
    target_ids = range(len(y))
    print(labels)
    labels = [lesion_type_dict[i] for i in labels]
    print(labels)
    plt.figure(figsize=(12, 10))
    colors = 'r', 'g', 'b', 'y', 'gray', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, labels):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label, alpha=.4)
    plt.legend(bbox_to_anchor=(.73, 1), loc=2, borderaxespad=0.) 
    print(k)
    if k==0:
        plt.ylim(-60, 60)
        plt.xlim(-60, 60)
    if k==1:
        plt.ylim(-60, 60)
        plt.xlim(-60, 60)
    if k==2:
        plt.ylim(-5, 4)
        plt.xlim(-4, 5)
    if k==3:
        plt.ylim(-4, 4)
        plt.xlim(-4, 4)
    if k==5:
        plt.ylim(-60, 60)
        plt.xlim(-60, 60)
    if k==4:
        plt.ylim(-60, 60)
        plt.xlim(-60, 60)
    if k==6:
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
    if k==7:
        plt.ylim(-11, 11)
        plt.xlim(-11, 11)
    if name is not None:
        plt.savefig('Graphs/'+name+'.png')
    plt.show()
#    shows T-SNE plot of linear/ convolutional trained model
#    cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')  
#    cluster.fit_predict(x)  
#    count = 0 
#    for i in range(len(ys)):
#        if cluster.labels_[i] == ys[i]:
#            count += 1
#    print(count/len(ys))  
    
def graphs():
    train_losses = []
    KLD_losses = []
    BCE_losses = []
    test_losses = []
    names = ["fcn mixture gaussian",
             "fcn resn mixture gaussian",
             "fcn mixture laplace",
             "fcn resn mixture laplace"]
    names_2 = ["fcn CE gaussian",
             "fcn resn CE gaussian",
             "fcn CE laplace",
             "fcn resn CE laplace"]
    models = ['models/v3/model_fcn_mixture_gaussian_100.pth',
              'models/v3/model_fcn_resn_mixture_gaussian_100.pth',
              'models/v3/model_fcn_mixture_laplace_100.pth',
              'models/v3/model_fcn_resn_mixture_laplace_100.pth']
    models_2 = ['models/v3/model_fcn_CE_gaussian_100.pth',
              'models/v3/model_fcn_resn_CE_gaussian_100.pth',
              'models/v3/model_fcn_CE_laplace_100.pth',
              'models/v3/model_fcn_resn_CE_laplace_100.pth']
#    kwargs = {'num_workers': 8, 'pin_memory': False}
#    path_test = 'test_dataset/'
#    test_loader = torch.utils.data.DataLoader(
#    data_loader.Xray_Dataset(path_test),
#    batch_size=64, shuffle=True, **kwargs)
#
#    print("Data is loaded")
#    i = 0
    for model in models:
        model, checkpoint = load_checkpoint(model)
        train_losses.append(np.array(checkpoint['train loss'])[:,0])
        KLD_losses.append(np.array(checkpoint['train loss'])[:,1])
        BCE_losses.append(np.array(checkpoint['train loss'])[:,2])
        test_losses.append(checkpoint['test loss'])
        #t-SNE plot
#        model.eval()
#        count = 0
#        with torch.no_grad():
#            for batch_idx, (data, label) in enumerate(test_loader):
#                z = model.get_z(data)
#                if count == 0:
#                    ys = label.numpy()
#                    zs = z.numpy()
#                else: 
#                    ys = np.append(ys, label.numpy())
#                    zs = np.append(zs, z.numpy(), axis=0)
#                count += 1
#        zs = zs.reshape(1001, 484) # 1001 or 2378
#        cluster_eval(zs, ys, 7, [0, 1, 2, 3, 4, 5, 6], i, names[i])
#        i+=1
       
    plt.figure(figsize=(12, 10))
    for i, (train_loss) in enumerate(train_losses):
        plt.plot(train_loss, label = names[i])
    plt.legend(bbox_to_anchor=(.752, 1), loc=2, borderaxespad=0.)   
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.savefig('Graphs/Train_1.png')
    plt.show()
    
    plt.figure(figsize=(12, 10))
    for i, (train_loss) in enumerate(KLD_losses):
        plt.plot(train_loss, label = names[i])
    plt.legend(bbox_to_anchor=(.752, 1), loc=2, borderaxespad=0.) 
    plt.xlabel('Epochs')
    plt.ylabel('KLD loss')
    plt.savefig('Graphs/KLD_1.png')
    plt.show()

    plt.figure(figsize=(12, 10))
    for i, (train_loss) in enumerate(BCE_losses):
        plt.plot(train_loss, label = names[i])
    plt.legend(bbox_to_anchor=(.752, 1), loc=2, borderaxespad=0.)  
    plt.xlabel('Epochs')
    plt.ylabel('Mixture loss')
    plt.savefig('Graphs/BCE_1.png')
    plt.show()

    plt.figure(figsize=(12, 10))
    for i, (train_loss) in enumerate(test_losses):
        plt.plot(train_loss, label = names[i])
    plt.legend(bbox_to_anchor=(.752, 1), loc=2, borderaxespad=0.)   
    plt.xlabel('Epochs')
    plt.ylabel('Test loss')
    plt.savefig('Graphs/Test_1.png')
    plt.show()    
    
#    
    i = 4
    train_losses = []
    KLD_losses = []
    BCE_losses = []
    test_losses = []
    for model in models_2:
        model, checkpoint = load_checkpoint(model)
        train_losses.append(np.array(checkpoint['train loss'])[:,0])
        KLD_losses.append(np.array(checkpoint['train loss'])[:,1])
        BCE_losses.append(np.array(checkpoint['train loss'])[:,2])
        test_losses.append(checkpoint['test loss'])
#        count = 0
#        with torch.no_grad():
#            for batch_idx, (data, label) in enumerate(test_loader):
#                z = model.get_z(data)
#                if count == 0:
#                    ys = label.numpy()
#                    zs = z.numpy()
#                else: 
#                    ys = np.append(ys, label.numpy())
#                    zs = np.append(zs, z.numpy(), axis=0)
#                count += 1
#        zs = zs.reshape(1001, 484) # 1001 or 2378
#        cluster_eval(zs, ys, 7, [0, 1, 2, 3, 4, 5, 6], i, names_2[i%4])
#        i+=1
#        
    plt.figure(figsize=(12, 10))
    for i, (train_loss) in enumerate(train_losses):
        plt.plot(train_loss, label = names_2[i])
    plt.legend(bbox_to_anchor=(.79, 1), loc=2, borderaxespad=0.) 
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.savefig('Graphs/Train_2.png')
    plt.show()
    
    plt.figure(figsize=(12, 10))
    for i, (train_loss) in enumerate(KLD_losses):
        plt.plot(train_loss, label = names_2[i])
    plt.legend(bbox_to_anchor=(.79, 1), loc=2, borderaxespad=0.) 
    plt.xlabel('Epochs')
    plt.ylabel('KLD loss')
    plt.savefig('Graphs/KLD_2.png')
    plt.show()

    plt.figure(figsize=(12, 10))
    for i, (train_loss) in enumerate(BCE_losses):
        plt.plot(train_loss, label = names_2[i])
    plt.legend(bbox_to_anchor=(.79, 1), loc=2, borderaxespad=0.)  
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy loss')
    plt.savefig('Graphs/BCE_2.png')
    plt.show()

    plt.figure(figsize=(12, 10))
    for i, (train_loss) in enumerate(test_losses):
        plt.plot(train_loss, label = names_2[i])
    plt.legend(bbox_to_anchor=(.79, 1), loc=2, borderaxespad=0.)   
    plt.xlabel('Epochs')
    plt.ylabel('Test loss')
    plt.savefig('Graphs/Test_2.png')
    plt.show()  
    


if __name__ == "__main__":
    graphs()
    model, checkpoint = load_checkpoint('models/v2/model_fcn_mixture_gaussian_100.pth')
    plt.plot(checkpoint['test loss'])
    plt.show()
    print("test loss")
    train_loss = np.array(checkpoint['train loss'])[:,0]
    plt.plot(train_loss)
    plt.show()
    print("train loss")
    train_loss = np.array(checkpoint['train loss'])[:,1]
    plt.plot(train_loss)
    plt.show()
    print("mixture")
    train_loss = np.array(checkpoint['train loss'])[:,2]
    plt.plot(train_loss)
    plt.show()
    print("KLD")
#    x_hats = checkpoint['test z']
#    ys = checkpoint['test label']
#    losses = checkpoint['loss']
    
#    print(testmodel)
#    
    kwargs = {'num_workers': 8, 'pin_memory': False}
    path_test = 'test_dataset/'
    test_loader = torch.utils.data.DataLoader(
    data_loader.Xray_Dataset(path_test),
    batch_size=64, shuffle=True, **kwargs)

    print("Data is loaded")
    model.eval()
    count = 0
    with torch.no_grad():
        print("starting")
        for batch_idx, (data, label) in enumerate(test_loader):
            z = model.get_z(data)

            if count == 0:
                ys = label.numpy()
                zs = z.numpy()
            else: 
                ys = np.append(ys, label.numpy())
                zs = np.append(zs, z.numpy(), axis=0)
            count += 1
            
    print(np.shape(zs))
    print(np.shape(ys))
    zs = zs.reshape(1001, 64) # 1001 or 2378
    cluster_eval(zs, ys, 7, [0, 1, 2, 3, 4, 5, 6])
    cluster_labels = split_labels(zs, ys)
    for i in range(6):
        for j in range(i+1,7):
            a = np.array(cluster_labels[i][0])
            b = np.array(cluster_labels[j][0])
            zs = np.append(a,b,axis=0)
            a = np.zeros(len(cluster_labels[i][1]))
            b = np.ones(len(cluster_labels[j][1]))
            ys = np.append(a,b,axis=0)
            cluster_eval(zs, ys, 2,[cluster_labels[i][1][0], cluster_labels[j][1][0]])
    
    