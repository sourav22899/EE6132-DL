import numpy as np
import matplotlib.pyplot as plt

from constants import *
from utils import *
from main import NeuralNet
from svm import SVM,KNN 

def learning_rates():
    alpha = np.logspace(-4,1,6)
    TrainLoss = np.zeros((6,N_ITERATIONS*5))
    TestLoss = np.zeros_like(TrainLoss)
    Accuracy = np.zeros_like(TrainLoss)

    for i in range(alpha.shape[0]):
        print('Learning Rate:',alpha[i])
        X,y,Xt,yt = load_data()
        net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation='sigmoid',lr=alpha[i])
        params = net.initialize().copy()
        y,TrainLoss[i],TestLoss[i],Accuracy[i] = net.train(X,y,Xt,yt,params)    

    np.save('lr_train_loss',TrainLoss)
    np.save('lr_test_loss',TestLoss)
    np.save('lr_accuracy',Accuracy)

    arr = np.load('lr_train_loss.npy')
    plt.figure(figsize=(18,9))
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Training Loss')
    X = np.arange(arr.shape[1])*200
    for i in range(arr.shape[0]):
        label_ = 'lr = '+ str(10**(i-4))
        plt.plot(X,arr[i],label=label_)
        plt.legend()

    plt.show()

# learning_rates()

def baseline_model(n=5,rate=LEARNING_RATE,l2_reg=0.0,activation='sigmoid',aug=False):
    if aug:
        TrainLoss = np.zeros((n,N_ITERATIONS*10))
    else:
        TrainLoss = np.zeros((n,N_ITERATIONS*5))
    TestLoss = np.zeros_like(TrainLoss)
    Accuracy = np.zeros_like(TrainLoss)

    for i in range(n):
        X,y,Xt,yt = load_data(aug=aug)
        net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate,reg=l2_reg)
        params = net.initialize().copy()
        y,TrainLoss[i],TestLoss[i],Accuracy[i] = net.train(X,y,Xt,yt,params)    

    train_loss = np.mean(TrainLoss,axis=0)
    test_loss = np.mean(TestLoss,axis=0)

    plt.figure(figsize=(18,9))
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    X = np.arange(train_loss.shape[0])*200
    plt.plot(X,train_loss,label='train_loss')
    plt.plot(X,test_loss,label='test_loss')
    plt.legend()

    plt.show()

# baseline_model(n=1,activation='sigmoid',rate=LEARNING_RATE,aug=True)

def baseline_accuracy(n=5,rate=LEARNING_RATE,l2_reg=0.0,activation='sigmoid',aug=False):
    accuracy_list = []
    for _ in range(n):
        X,y,Xt,yt = load_data(aug=aug)
        net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate)
        params = net.initialize().copy()
        accuracy_list.append(net.train(X,y,Xt,yt,params,metrics=True))

    accuracy_list = np.asarray(accuracy_list)
    print('Accuracy:',np.mean(accuracy_list))
    print('Standard Deviation:',np.std(accuracy_list))   

# baseline_accuracy(n=3,aug=True)
# baseline_model(n=1,activation='sigmoid',rate=LEARNING_RATE)


def baseline_inactive(rate=LEARNING_RATE,l2_reg=0.0,activation='sigmoid'):
    X,y,Xt,yt = load_data()
    net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate,reg=l2_reg)
    params = net.initialize().copy()
    return net.train(X,y,Xt,yt,params,inactive=True)    

def baseline_inactive_compared():
    sig = baseline_inactive(activation='sigmoid')
    tan = baseline_inactive(activation='tanh')
    rel = baseline_inactive(activation='relu')

    plt.figure(figsize=(18,9))
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Percentage of inactive neurons')
    X = np.arange(len(tan))*200
    plt.plot(X,sig,label='sigmoid')
    plt.plot(X,tan,label='tanh')
    plt.plot(X,rel,label='relu')
    plt.legend()
    plt.show()

# baseline_inactive_compared()

def l2_regularization(n=5,rate=LEARNING_RATE,l2_reg=0.0,activation='sigmoid',n_iter=N_ITERATIONS):
    TrainLoss = np.zeros((n,n_iter*5))
    TestLoss = np.zeros_like(TrainLoss)
    Accuracy = np.zeros_like(TrainLoss)

    for i in range(n):
        X,y,Xt,yt = load_data()
        net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate,reg=0.0)
        params = net.initialize().copy()
        y,TrainLoss[i],TestLoss[i],Accuracy[i] = net.train(X,y,Xt,yt,params,n_iter=n_iter)    


    train_loss = np.mean(TrainLoss,axis=0)
    test_loss = np.mean(TestLoss,axis=0)

    for i in range(n):
        X,y,Xt,yt = load_data()
        net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate,reg=l2_reg)
        params = net.initialize().copy()
        y,TrainLoss[i],TestLoss[i],Accuracy[i] = net.train(X,y,Xt,yt,params,n_iter=n_iter)    


    train_loss_reg = np.mean(TrainLoss,axis=0)
    test_loss_reg = np.mean(TestLoss,axis=0)

    fig = plt.figure(figsize=(18,9))
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Training Loss')
    X = np.arange(train_loss.shape[0])*200
    plt.plot(X,train_loss,label='train_loss')
    plt.plot(X,train_loss_reg,label='train_loss_reg')
    plt.legend()
    plt.show()


    fig = plt.figure(figsize=(18,9))
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Test Loss')
    X = np.arange(train_loss.shape[0])*200
    plt.plot(X,test_loss,label='test_loss')
    plt.plot(X,test_loss_reg,label='test_loss_reg')
    plt.legend()
    plt.show()


def baseline_noisy(n=5,rate=LEARNING_RATE,l2_reg=0.0,activation='sigmoid',fwd_std=0.0,bkd_std=0.0):
    TrainLoss = np.zeros((n,N_ITERATIONS*5))
    TestLoss = np.zeros_like(TrainLoss)
    Accuracy = np.zeros_like(TrainLoss)

    for i in range(n):
        X,y,Xt,yt = load_data()
        net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate,reg=l2_reg,fwd_std=0.0,bkd_std=0.0)
        params = net.initialize().copy()
        y,TrainLoss[i],TestLoss[i],Accuracy[i] = net.train(X,y,Xt,yt,params)   

    train_loss = np.mean(TrainLoss,axis=0)
    test_loss = np.mean(TestLoss,axis=0)

    for i in range(n):
        X,y,Xt,yt = load_data()
        net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate,reg=l2_reg,fwd_std=fwd_std,bkd_std=0.0)
        params = net.initialize().copy()
        y,TrainLoss[i],TestLoss[i],Accuracy[i] = net.train(X,y,Xt,yt,params)    

    train_loss_fwd = np.mean(TrainLoss,axis=0)
    test_loss_fwd = np.mean(TestLoss,axis=0)

    for i in range(n):
        X,y,Xt,yt = load_data()
        net = NeuralNet(n_hidden=len(L),layers=L,input_size=INPUT_SIZE,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate,reg=l2_reg,fwd_std=0.0,bkd_std=bkd_std)
        params = net.initialize().copy()
        y,TrainLoss[i],TestLoss[i],Accuracy[i] = net.train(X,y,Xt,yt,params)    

    train_loss_bkd = np.mean(TrainLoss,axis=0)
    test_loss_bkd = np.mean(TestLoss,axis=0)

    plt.figure(figsize=(18,9))
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Test Loss')
    X = np.arange(train_loss.shape[0])*200
    plt.plot(X,test_loss,label='test_loss')
    plt.plot(X,test_loss_fwd,label='test_loss_fwd')
    plt.plot(X,test_loss_bkd,label='test_loss_bkd')
    plt.legend()

    plt.show()

    plt.figure(figsize=(18,9))
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Train Loss')
    X = np.arange(train_loss.shape[0])*200
    plt.plot(X,train_loss,label='train_loss')
    plt.plot(X,train_loss_fwd,label='train_loss_fwd')
    plt.plot(X,train_loss_bkd,label='train_loss_bkd')
    plt.legend()

    plt.show()

# baseline_noisy(n=3,fwd_std=1,bkd_std=1)

def hog_model(n=5,rate=LEARNING_RATE,l2_reg=0.0,activation='relu',aug=False,hog=True):
    if aug:
        TrainLoss = np.zeros((n,N_ITERATIONS*10))
    else:
        TrainLoss = np.zeros((n,N_ITERATIONS*5))
    TestLoss = np.zeros_like(TrainLoss)
    Accuracy = np.zeros_like(TrainLoss)

    for i in range(n):
        X,y,Xt,yt = load_data(aug=aug,hog=True)
        X = X.T;Xt = Xt.T
        net = NeuralNet(n_hidden=len(L_HOG),layers=L_HOG,input_size=INPUT_SIZE_HOG,n_classes=NUM_CLASSES,batch_size=BATCH_SIZE,activation=activation,lr=rate,reg=l2_reg)
        params = net.initialize().copy()
        y,TrainLoss[i],TestLoss[i],Accuracy[i] = net.train(X,y,Xt,yt,params)    

    train_loss = np.mean(TrainLoss,axis=0)
    test_loss = np.mean(TestLoss,axis=0)

    plt.figure(figsize=(18,9))
    plt.grid()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    X = np.arange(train_loss.shape[0])*200
    plt.plot(X,train_loss,label='train_loss')
    plt.plot(X,test_loss,label='test_loss')
    plt.legend()

    plt.show()

# hog_model(n=3,rate=LEARNING_RATE,activation='relu')
# l2_regularization(n=3,rate=LEARNING_RATE,l2_reg=REGULARIZATION,activation='sigmoid',n_iter=N_ITERATIONS)


def print_instructions():
    print('==================================================================================')
    print('Before running any codes, please check requirements.txt to make sure that all the needed libraries are installed.')
    print('You may also execute $python3 train.py in the terminal to execute the codes.')
    print('Enter the number corresponding to the questions to execute code for that particular question.')
    print('0 -- Exit.')   
    print('1 -- Baseline training.')
    print('2 -- Varying learning rates.')
    print('3 -- Detailed baseline accuracy metrics.')
    print('4 -- Baseline with tanh activation.')
    print('5 -- Baseline with ReLU activation.')
    print('6 -- Compare fraction of inactive neurons.')
    print('7 -- Add noise to forward and backward pass.')
    print('8 -- Use data augmentation.')
    print('9 -- L2-Regularization.')
    print('10 -- Train neural network on HOG features.')
    print('11 -- Train KNN/SVM on HOG features.')
    print('==================================================================================')

def select(x):
    if x == 1:
        baseline_model(n=1)
    elif x == 2:
        learning_rates()
    elif x == 3:
        baseline_accuracy(n=1)
    elif x == 4:
        baseline_model(n=1,activation='tanh')
    elif x == 5:
        baseline_model(n=1,activation='relu')
    elif x == 6:
        baseline_inactive_compared()
    elif x == 7:
        baseline_noisy(n=1,fwd_std=FWD_STD_DEV,bkd_std=BKD_STD_DEV)
    elif x == 8:
        baseline_model(n=1,aug=True)
    elif x == 9:
        l2_regularization(n=1)
    elif x == 10:
        hog_model(n=1,activation='relu')
    elif x == 11:
        SVM()
        KNN()
    else:
        print('Invalid input.')

print_instructions()
x = int(input('Enter a value from 1 to 11 (0 to exit):'))
while x:
    select(x)
    print_instructions()
    x = int(input('Enter a value from 1 to 11 (0 to exit):'))