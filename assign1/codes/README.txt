This directory contains the following files in the tree structure:
.
├── codes
│   ├── assignment1.pdf
│   ├── constants.py
│   ├── download.py
│   ├── hog_features.py
│   ├── knn_svm_results.txt
│   ├── logs.txt
│   ├── main.py
│   ├── preprocessing.py
│   ├── README.txt
│   ├── requirements.txt
│   ├── run.sh
│   ├── svm.py
│   ├── train.py
│   └── utils.py
└── MNIST


###############################################################################################
        *** Please maintain the directory structure for smooth running of the codes ***
###############################################################################################

- assignment1.pdf -- pdf report of the assignment
- constants.py -- Contains all the hyperparameters.
- download.py -- Contains the code to download the dataset from http://yann.lecun.com/exdb/mnist/.
                 [1] https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03
- hog_features.py -- Extracts the HOG features and stores it in numpy arrays and stores in /MNIST.
- knn_svm_results.txt -- Contains the final results of the KNN and SVM classifier.
- logs.txt -- Contains the logging information of the training process of the neural net.
- main.py -- Contains the class NeuralNet() and the required methods.
- preprocessing.py -- Converts the dataset downloaded from http://yann.lecun.com/exdb/mnist/ into numpy                           arrays and stores in /MNIST. It also simultaneously generates the augmented dataset                          containing the noisy data.
- requirements.txt -- Contains all the libraries needed to run the codes.
                      If needed, execute $pip install -r requirements.txt 

- run.sh -- *** Download and convert the given dataset into numpy arrays and execute the questions. ***
                $chmod +x run.sh
                $./run.sh

- svm.py -- Contains the functions to implement SVM as well as KNN.
- train.py -- Contains all the functions to implement the multilayer perceptron and the additional parts                   mentioned in the assignment.
- utils.py -- Contains all the utility functions.

- MNIST -- If dataset are available simply paste the idx-3-ubyte files in this directory.All the numpy data             arrays generated will be stored here.