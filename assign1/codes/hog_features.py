import numpy as np
from skimage.feature import hog

print('Extracting HOG Features...')
print('This step takes 40-60 seconds...')

train_images = np.load('../MNIST/train_images.npy')
test_images = np.load('../MNIST/test_images.npy')

train_images_hog,test_images_hog = [],[]
for i in range(train_images.shape[1]):
    sample = train_images[:,i].reshape(28,28)
    train_images_hog.append(hog(sample,orientations=9,cells_per_block=(2,2),pixels_per_cell=(7,7),visualize=False))

for i in range(test_images.shape[1]):
    sample = test_images[:,i].reshape(28,28)
    test_images_hog.append(hog(sample,orientations=9,cells_per_block=(2,2),pixels_per_cell=(7,7),visualize=False))

np.save('../MNIST/train_images_hog',train_images_hog)
np.save('../MNIST/test_images_hog',test_images_hog)