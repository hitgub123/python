import torch as tc, numpy as np, pandas as pd, os
from sklearn import preprocessing
from sklearn.datasets import fetch_openml

#####################################################
#################### data ##########################
#####################################################
mnist = fetch_openml("mnist_784", version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)
X, y = mnist["data"], mnist["target"]
X[:60000].to_numpy()
X_train, X_test, y_train, y_test = map(
    lambda df: df.to_numpy(), (X[:60000], X[60000:], y[:60000], y[60000:])
)

# 洗牌训练集
np.random.seed(666)
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
X_train = preprocessing.StandardScaler().fit_transform(X_train)
X_test = preprocessing.StandardScaler().fit_transform(X_test)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
X_train, y_train, X_test, y_test = map(tc.tensor, (X_train, y_train, X_test, y_test))
X_train, X_test = X_train.to(tc.float64), X_test.to(tc.float64)
y_train, y_test = y_train.to(tc.long), y_test.to(tc.long)
y_train = tc.nn.functional.one_hot(y_train, num_classes=10)
y_train = y_train.to(tc.float64)


#####################################################
#################### model ##########################
#####################################################
hidden_size1 = 128
hidden_size2 = 70
data_size = X_train.shape[0]
input_size = X_train.shape[-1]
output_size = 10
batch_size = 128
lr = 0.001
epochs = 200


model_name = "pytorch/learn1/models/{}".format("002_mnist预测_01_FC.pkl")

model = tc.nn.Sequential(
    tc.nn.Linear(input_size, hidden_size1, dtype=float),
    tc.nn.ReLU(),
    tc.nn.Linear(hidden_size1, hidden_size2, dtype=float),
    tc.nn.ReLU(),
    tc.nn.Linear(hidden_size2, output_size, dtype=float),
)
cost = tc.nn.CrossEntropyLoss()
optimizer = tc.optim.Adam(model.parameters(), lr=lr)


def test():
    d=tc.load("pytorch/learn1/models/{}".format("test.pkl"))
    model.load_state_dict(d['w'])
    prediction = model(X_test).argmax(axis=1)
    acc_rate = (prediction == y_test).sum() / y_test.size()[0]
    print(acc_rate.item())  # 0.9789999723434448
    for i,j in enumerate(model.named_parameters()):
        print(i,j)
    # tensor([-0.1227, -0.1915, -0.1522, -0.0182,  0.1966, -0.1371, -0.1437, -0.1129,
    #     0.4061,  0.0867], dtype=torch.float64, requires_grad=True))


if __name__ == "__main__":
    test()
