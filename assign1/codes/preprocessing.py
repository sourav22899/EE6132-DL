import numpy as np
import idx2numpy
import cv2

from constants import *

files = ['../MNIST/t10k-images.idx3-ubyte','../MNIST/t10k-labels.idx1-ubyte',
        '../MNIST/train-images.idx3-ubyte','../MNIST/train-labels.idx1-ubyte']

def sanity_check():
    images = np.load('../MNIST/train_images.npy')
    labels = np.load('../MNIST/train_labels.npy')
    x = np.random.randint(100)
    cv2.imshow('Image',images[:,x].reshape(28,28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(labels[:,x])

def normalize(x):
    x = x.reshape(x.shape[0],-1)
    x = x/255.0
    return x.T

def one_hot(x):
    y = np.zeros((NUM_CLASSES,x.shape[0]))
    t = np.arange(x.shape[0])
    y[x,t] = 1
    return y

print('Converting idx file type into numpy ndarrays...')
print('This step takes a few seconds...')

for f in files:
    arr = idx2numpy.convert_from_file(f)
    if f == '../MNIST/t10k-images.idx3-ubyte':
        arr = normalize(arr)
        np.save('../MNIST/test_images',arr)
    elif f == '../MNIST/t10k-labels.idx1-ubyte':
        arr = one_hot(arr)
        np.save('../MNIST/test_labels',arr)
    elif f == '../MNIST/train-images.idx3-ubyte':
        np.save('../MNIST/train_images',normalize(arr))
        noise = np.random.randn(*arr.shape)*NOISE_STD_DEV
        arr = arr + noise
        arr = normalize(arr)
        np.save('../MNIST/train_images_noise',arr)
    elif f == '../MNIST/train-labels.idx1-ubyte':
        arr = one_hot(arr)
        np.save('../MNIST/train_labels',arr)
    print(arr.shape)

a = np.load('../MNIST/train_images_noise.npy')
b = np.load('../MNIST/train_images.npy')
np.save('../MNIST/train_images_aug',np.concatenate((a,b),axis=1))
b = np.load('../MNIST/train_labels.npy')
np.save('../MNIST/train_labels_aug',np.concatenate((b,b),axis=1))

# sanity_check()
