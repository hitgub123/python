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

X_train, y_train, X_test, y_test = map(tc.tensor, (X_train, y_train, X_test, y_test))
X_train, X_test = X_train.to(tc.float64), X_test.to(tc.float64)
X_train, X_test = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)
y_train, y_test = y_train.to(tc.long), y_test.to(tc.long)
y_train = tc.nn.functional.one_hot(y_train, num_classes=10)
y_train = y_train.to(tc.float64)


#####################################################
#################### model ##########################
#####################################################

data_size = X_train.shape[0]
output_size = 10
batch_size = 128
lr = 0.001
epochs = 30
out_channels1 = 16
out_channels2 = 32
kernel_size = 5
stride = 1
padding = 2

model_name = os.path.basename(__file__).replace(".py", ".pkl")
model_name = "pytorch/learn1/models/{}".format(model_name)

model = tc.nn.Sequential(
    tc.nn.Conv2d(  # in:1*28*28,out:16*28*28
        in_channels=1,
        out_channels=out_channels1,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dtype=float,
    ),
    tc.nn.ReLU(),
    tc.nn.MaxPool2d(kernel_size=2),
    tc.nn.Conv2d(  # in:16*14*14,out:32*14*14
        out_channels1, out_channels2, kernel_size, stride, padding, dtype=float
    ),
    tc.nn.ReLU(),
    tc.nn.MaxPool2d(2),
    tc.nn.Flatten(),
    tc.nn.Linear(out_channels2 * 7 * 7, output_size, dtype=float),
)
cost = tc.nn.CrossEntropyLoss()
optimizer = tc.optim.Adam(model.parameters(), lr=lr)


def train():
    for i in range(epochs):
        batch_loss = []
        for start in range(0, data_size, batch_size):
            optimizer.zero_grad()
            end = start + batch_size if start + batch_size < data_size else data_size
            x, y = X_train[start:end], y_train[start:end]
            prediction = model(x)
            loss = cost(prediction, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        if not i % 10:
            print(i, tc.tensor(batch_loss).mean().item())
            print(
                "current accuracy",
                (model(X_test).argmax(axis=1) == y_test).sum() / y_test.size()[0],
            )

    tc.save(model.state_dict(), model_name)

#####################################################
#################### test ##########################
#####################################################
def test():
    model.load_state_dict(tc.load(model_name, weights_only=0))
    prediction = model(X_test).argmax(axis=1)
    acc_rate = (prediction == y_test).sum() / y_test.size()[0]
    print(acc_rate.item())
    # print(model)
    # for i,j in enumerate(model.named_parameters()):
    #     print(i,j)


if __name__ == "__main__":
    # train()
    test()
