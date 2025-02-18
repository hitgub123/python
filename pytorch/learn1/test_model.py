import torch as tc, numpy as np, pandas as pd
from sklearn import preprocessing
from sklearn.datasets import fetch_openml

#####################################################
#################### load model ##########################
#####################################################
path = "pytorch/learn1/models/"
model_name = path+"mnist.pkl"

#####################################################
#################### data ##########################
#####################################################
mnist = fetch_openml("mnist_784", version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = map(
    lambda df: df.to_numpy(), (X[:60000], X[60000:], y[:60000], y[60000:])
)

X_test = preprocessing.StandardScaler().fit_transform(X_test)
X_test, y_test = map(tc.tensor, (X_test, y_test))
X_test = X_test.to(tc.float64)
y_test = y_test.to(tc.long)

#####################################################
#################### model ##########################
#####################################################
hidden_size = 128
data_size = 60000
input_size = 784
output_size = 10
batch_size = 128
lr = 0.001
epochs = 100
model = tc.nn.Sequential(
    tc.nn.Linear(input_size, hidden_size, dtype=float),
    tc.nn.ReLU(),
    # tc.nn.Sigmoid(),
    tc.nn.Linear(hidden_size, output_size, dtype=float),
)
model.load_state_dict(tc.load(model_name, weights_only=0))
prediction = model(X_test).argmax(axis=1)
acc_rate = (prediction == y_test).sum() / y_test.size()[0]
print(acc_rate.item())
