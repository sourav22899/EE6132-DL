import numpy as np
import matplotlib.pyplot as plt

from constants import *
from random import shuffle

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relu_grad(x):
    return (x>0)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - tanh(x)*tanh(x)

def softmax(x):
    y = np.exp(x - np.max(x,axis=0)) 
    return y/np.sum(y,axis=0) 

def ce_loss(y,y_hat,epsilon=1e-8,BATCH_SIZE=64):
    return -np.sum(y*np.log(y_hat))
    
def glorot_initialization(x):
    assert len(x.shape) == 2
    limit = np.sqrt(6/(x.shape[0]+x.shape[1]))
    return np.random.uniform(-limit,limit,x.shape)

def load_data(aug=False,hog=False):
    if aug:
        X = np.load('../MNIST/train_images_aug.npy')
        y = np.load('../MNIST/train_labels_aug.npy')
        Xt = np.load('../MNIST/test_images.npy')
        yt = np.load('../MNIST/test_labels.npy')
    elif hog:
        X = np.load('../MNIST/train_images_hog.npy')
        Xt = np.load('../MNIST/test_images_hog.npy')
        yt = np.load('../MNIST/test_labels.npy')
        y = np.load('../MNIST/train_labels.npy')
    else:
        X = np.load('../MNIST/train_images.npy')
        y = np.load('../MNIST/train_labels.npy')
        Xt = np.load('../MNIST/test_images.npy')
        yt = np.load('../MNIST/test_labels.npy')
    return X,y,Xt,yt

def create_batches(X,y):
    idx = [i for i in range(X.shape[1])]
    shuffle(idx)
    X_ = X[:,idx];y_ = y[:,idx]
    n_batches = int(X.shape[1]/BATCH_SIZE)
    batch_X,batch_y = [],[]
    for i in range(n_batches):
        batch_X.append(X_[:,i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        batch_y.append(y_[:,i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    
    return np.asarray(batch_X),np.asarray(batch_y),n_batches

def plot_confusion_matrix(y,y_pred,n_class=NUM_CLASSES):
    hist = np.zeros((n_class,n_class))
    for i,j in zip(y,y_pred):
        hist[i,j] += 1
    return hist

def inactive_neurons(grads,err=1e-5):
    n_neurons,n_inactive_neurons = 0,0
    for key in list(grads.keys()):
        if key[0] == 'z':
            n_neurons += (grads[key].shape[0]*grads[key].shape[1])
            temp = np.abs(grads[key]) < err
            assert grads[key].shape[0] == temp.shape[0]
            n_inactive_neurons += np.sum(temp)
    
    return 100*float(n_inactive_neurons/n_neurons)
