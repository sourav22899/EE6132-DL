import numpy as np

from constants import *
from utils import *

class NeuralNet():
    def __init__(self,n_hidden,layers,input_size,n_classes,batch_size,activation=None,lr=LEARNING_RATE,reg=0.0,fwd_std=0.0,bkd_std=0.0):
        self.hidden = n_hidden
        self.layers = layers
        self.input_size = input_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.activation = activation
        self.lr = lr
        self.reg = reg
        self.fwd_std = fwd_std
        self.bkd_std = bkd_std
            
    def initialize(self):
        layers = self.layers.copy()
        layers.insert(0,self.input_size)
        layers.append(self.n_classes)
        params = {}
        for i in range(len(layers)-1):
            w = np.zeros((layers[i+1],layers[i]))
            params['W'+str(i)] = glorot_initialization(w)
            params['b'+str(i)] = np.zeros(layers[i+1])[:,np.newaxis]

        return params

    def forward_prop(self,x,params):
        params['a0'] = x # x.copy()
        for i in range(self.hidden):
            params['z'+str(i)] = params['W'+str(i)].dot(params['a'+str(i)]) + params['b'+str(i)]
            params['z'+str(i)] += (np.random.randn(*params['z'+str(i)].shape))*self.fwd_std
            if self.activation == 'sigmoid':
                params['a'+str(i+1)] = sigmoid(params['z'+str(i)])
            elif self.activation == 'relu':
                params['a'+str(i+1)] = relu(params['z'+str(i)])
            elif self.activation == 'tanh':
                params['a'+str(i+1)] = tanh(params['z'+str(i)])
        params['z'+str(self.hidden)] = params['W'+str(self.hidden)].dot(params['a'+str(self.hidden)]) + params['b'+str(self.hidden)]
        y_hat = softmax(params['z'+str(self.hidden)])       

        return params,y_hat

    def loss(self,y,y_hat,params):
        reg_loss = 0.0
        for key in list(params.keys()):
            if key[0] == 'W':
                reg_loss += self.reg*np.sum(np.square(params[key]))
        return ce_loss(y,y_hat) + reg_loss

    def backward_prop_sigmoid(self,y,y_hat,params):
        grads = {}

        grads['z3'] = np.random.randn(*y.shape)*self.bkd_std
        grads['z3'] = grads['z3'] + (y_hat - y)/BATCH_SIZE
        grads['W3'] = grads['z3'].dot(params['a3'].T) + 2*self.reg*params['W3']
        grads['b3'] = np.sum(grads['z3'],axis=1,keepdims=True)
        grads['a3'] = params['W3'].T.dot(grads['z3'])

        grads['z2'] = grads['a3']*sigmoid_grad(params['z2'])
        grads['W2'] = grads['z2'].dot(params['a2'].T) + 2*self.reg*params['W2']
        grads['b2'] = np.sum(grads['z2'],axis=1,keepdims=True)
        grads['a2'] = params['W2'].T.dot(grads['z2'])

        grads['z1'] = grads['a2']*sigmoid_grad(params['z1'])
        grads['W1'] = grads['z1'].dot(params['a1'].T) + 2*self.reg*params['W1']
        grads['b1'] = np.sum(grads['z1'],axis=1,keepdims=True)
        grads['a1'] = params['W1'].T.dot(grads['z1'])

        grads['z0'] = grads['a1']*sigmoid_grad(params['z0'])
        grads['W0'] = grads['z0'].dot(params['a0'].T) + 2*self.reg*params['W0']
        grads['b0'] = np.sum(grads['z0'],axis=1,keepdims=True)

        return grads,params

    def backward_prop_relu(self,y,y_hat,params):
        grads = {}
        
        grads['z3'] = (y_hat - y)/BATCH_SIZE
        grads['W3'] = grads['z3'].dot(params['a3'].T) + 2*self.reg*params['W3']
        grads['b3'] = np.sum(grads['z3'],axis=1,keepdims=True)
        grads['a3'] = params['W3'].T.dot(grads['z3'])

        grads['z2'] = grads['a3']*relu_grad(params['z2'])
        grads['W2'] = grads['z2'].dot(params['a2'].T) + 2*self.reg*params['W2']
        grads['b2'] = np.sum(grads['z2'],axis=1,keepdims=True)
        grads['a2'] = params['W2'].T.dot(grads['z2'])

        grads['z1'] = grads['a2']*relu_grad(params['z1'])
        grads['W1'] = grads['z1'].dot(params['a1'].T) + 2*self.reg*params['W1']
        grads['b1'] = np.sum(grads['z1'],axis=1,keepdims=True)
        grads['a1'] = params['W1'].T.dot(grads['z1'])

        grads['z0'] = grads['a1']*relu_grad(params['z0'])
        grads['W0'] = grads['z0'].dot(params['a0'].T) + 2*self.reg*params['W0']
        grads['b0'] = np.sum(grads['z0'],axis=1,keepdims=True)

        return grads,params

    def backward_prop_tanh(self,y,y_hat,params):
        grads = {}
        
        grads['z3'] = (y_hat - y)/BATCH_SIZE
        grads['W3'] = grads['z3'].dot(params['a3'].T) + 2*self.reg*params['W3']
        grads['b3'] = np.sum(grads['z3'],axis=1,keepdims=True)
        grads['a3'] = params['W3'].T.dot(grads['z3'])

        grads['z2'] = grads['a3']*tanh_grad(params['z2'])
        grads['W2'] = grads['z2'].dot(params['a2'].T) + 2*self.reg*params['W2']
        grads['b2'] = np.sum(grads['z2'],axis=1,keepdims=True)
        grads['a2'] = params['W2'].T.dot(grads['z2'])

        grads['z1'] = grads['a2']*tanh_grad(params['z1'])
        grads['W1'] = grads['z1'].dot(params['a1'].T) + 2*self.reg*params['W1']
        grads['b1'] = np.sum(grads['z1'],axis=1,keepdims=True)
        grads['a1'] = params['W1'].T.dot(grads['z1'])

        grads['z0'] = grads['a1']*tanh_grad(params['z0'])
        grads['W0'] = grads['z0'].dot(params['a0'].T) + 2*self.reg*params['W0']
        grads['b0'] = np.sum(grads['z0'],axis=1,keepdims=True)

        return grads,params

    def gradient_descent(self,params,grads,alpha):
        for key in list(grads.keys()):
            params[key] = params[key] - alpha*grads[key]
        
        return params,grads


    def train(self,X,y,Xt,yt,params,n_iter=N_ITERATIONS,metrics=False,inactive=False):
        TrainLoss,TestLoss,Accuracy,Inactive = [],[],[],[]
        for i in range(n_iter):
            X_train,y_train,n_batches = create_batches(X,y)
            loss = 0.0
            for j in range(n_batches): 
                _,y_hat = self.forward_prop(X_train[j],params)
                loss = loss + self.loss(y_train[j],y_hat,params)
                if self.activation == 'sigmoid':
                    grads,params_temp = self.backward_prop_sigmoid(y_train[j],y_hat,params)
                elif self.activation == 'tanh':
                    grads,params_temp = self.backward_prop_tanh(y_train[j],y_hat,params)                   
                elif self.activation == 'relu':
                    grads,params_temp = self.backward_prop_relu(y_train[j],y_hat,params)
                params,_ = self.gradient_descent(params_temp,grads,self.lr)            
            
                loss = loss/BATCH_SIZE
                if j % 200 == 0:
                    test_loss,acc = self.predict(Xt,yt,params)
                    print('Epoch:',i+1,'Iter:',j,'Train Loss:',loss,'Test Loss:',test_loss,
                                'Test Acc:',acc)
                    TrainLoss.append(loss)
                    TestLoss.append(test_loss)
                    Accuracy.append(acc)
                    if inactive:
                        Inactive.append(inactive_neurons(grads))
        
        if inactive:
            return Inactive
            
        if metrics:
            _,accuracy,precision,recall,f1,hist = self.accuracy_results(Xt,yt,params)
            print('Accuracy:',accuracy)
            print('Precision:',precision)
            print('Recall:',recall)
            print('F1-Scores:',f1)
            print('Confusion Matrix:',hist)

            return accuracy

        return params,TrainLoss,TestLoss,Accuracy
   
    def predict(self,X,y,params):
        X_test,y_test,n_batches = create_batches(X,y)
        nc,loss = 0,0.0
        for i in range(n_batches):
            x = X_test[i]
            for j in range(self.hidden):
                x = params['W'+str(j)].dot(x) + params['b'+str(j)]
                if self.activation == 'sigmoid':
                    x = sigmoid(x)
                elif self.activation == 'relu':
                    x = relu(x)
                elif self.activation == 'tanh':
                    x = tanh(x)
            x = params['W'+str(self.hidden)].dot(x) + params['b'+str(self.hidden)]
            preds = softmax(x)       

            # _,preds = self.forward_prop(X_test[i],params)
            y_pred = np.argmax(preds,axis=0)
            y_argmax = np.argmax(y_test[i],axis=0)
            loss += ce_loss(y_test[i],preds)
            for p,q in zip(y_argmax,y_pred):
                if p == q:
                    nc += 1
        accuracy = nc/(n_batches*BATCH_SIZE)
        loss = loss/(n_batches*BATCH_SIZE)
        return loss,accuracy

    def accuracy_results(self,X,y,params):
        X_test,y_test,n_batches = create_batches(X,y)
        nc,loss = 0,0.0
        Y_preds,Y = [],[]
        for i in range(n_batches):
            _,preds = self.forward_prop(X_test[i],params)
            y_pred = np.argmax(preds,axis=0)
            y_argmax = np.argmax(y_test[i],axis=0)
            Y_preds.extend(y_pred)
            Y.extend(y_argmax)
            loss += ce_loss(y_test[i],preds)
            for p,q in zip(y_argmax,y_pred):
                if p == q:
                    nc += 1
        accuracy = nc/(n_batches*BATCH_SIZE)
        loss = loss/(n_batches*BATCH_SIZE)
        hist = plot_confusion_matrix(Y,Y_preds)

        p = np.sum(hist,axis=0,keepdims=True)
        precision = np.diag(hist/p).flatten()

        r = np.sum(hist,axis=1,keepdims=True)
        recall = np.diag(hist/r).flatten()

        f1 = np.divide(2*precision*recall,precision+recall)
        
        hist = np.asarray(hist,dtype=np.int32)
        return loss,accuracy,precision,recall,f1,hist
