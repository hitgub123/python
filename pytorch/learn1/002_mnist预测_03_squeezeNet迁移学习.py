import torch as tc, numpy as np, pandas as pd, os
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from torchvision import models, transforms, datasets

#####################################################
#################### data ##########################
#####################################################
mnist = fetch_openml("mnist_784", version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)
X, y = mnist["data"], mnist["target"]
train_data_size = 60000  # 60000
X[:train_data_size].to_numpy()
X_train, X_test, y_train, y_test = map(
    lambda df: df.to_numpy(),
    (X[:train_data_size], X[60000:], y[:train_data_size], y[60000:]),
)

# 洗牌训练集
np.random.seed(666)
shuffle_index = np.random.permutation(train_data_size)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
X_train = preprocessing.StandardScaler().fit_transform(X_train)
X_test = preprocessing.StandardScaler().fit_transform(X_test)

X_train, y_train, X_test, y_test = map(tc.tensor, (X_train, y_train, X_test, y_test))
X_train, X_test = X_train.to(tc.float), X_test.to(tc.float)
X_train, X_test = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)
y_train, y_test = y_train.to(tc.long), y_test.to(tc.long)
y_train = tc.nn.functional.one_hot(y_train, num_classes=10)
y_train = y_train.to(tc.float)


def transform_input(input_tensor):
    padded_tensor = tc.nn.functional.pad(
        input_tensor, (98, 98, 98, 98), mode="constant", value=0
    )
    output_tensor = padded_tensor.repeat(1, 3, 1, 1)
    return output_tensor


X_test = transform_input(X_test)

#####################################################
#################### model ##########################
#####################################################

model_name = os.path.basename(__file__).replace(".py", ".pkl")
model_name = "pytorch/learn1/models/{}".format(model_name)
# model = models.squeezenet1_(weights=models.squeezenet.SqueezeNet1_0_Weights.DEFAULT)
model = models.squeezenet1_1(weights=models.squeezenet.SqueezeNet1_1_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False
model.classifier = tc.nn.Sequential(tc.nn.Flatten(), tc.nn.Linear(13 * 13 * 512, 10))
# model.load_state_dict(tc.load(model_name))
if os.path.exists(model_name):
    model.classifier.load_state_dict(tc.load(model_name))
cost = tc.nn.CrossEntropyLoss()
optimizer = tc.optim.Adam(model.classifier.parameters(), lr=0.001)

data_size = X_train.shape[0]
batch_size = 32

print(model)


def train():
    max_rate = 0.9425
    # max_rate = 0
    for i in range(100):
        batch_loss = []
        for start in range(0, data_size, batch_size):
            optimizer.zero_grad()
            end = start + batch_size if start + batch_size < data_size else data_size
            x, y = X_train[start:end], y_train[start:end]
            x = transform_input(x)
            prediction = model(x)
            loss = cost(prediction, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        if not i % 1:
            new_rate = test()
            print(i, tc.tensor(batch_loss).mean().item(), "acc rate=", new_rate)
            if new_rate > max_rate:
                max_rate = new_rate
                tc.save(model.classifier.state_dict(), model_name)

    # tc.save(model.state_dict(), model_name)
    # tc.save(model.classifier.state_dict(), model_name)
    # tc.save({'max_rate':max_rate,'W':model.state_dict()}, model_name)


#####################################################
#################### test ##########################
#####################################################
def test():
    # model.load_state_dict(tc.load(model_name, weights_only=0))
    # if os.path.exists(model_name):
    # model.classifier.load_state_dict(torch.load(model_name))
    acc_count = 0
    epoch = 2
    batch_size = 1000
    for i in range(epoch):
        prediction = model(
            X_test[batch_size * i : batch_size * (i + 1)].view(-1, 3, 224, 224)
        ).argmax(axis=1)
        acc_count += (y_test[batch_size * i : batch_size * (i + 1)] == prediction).sum()
    acc_rate = acc_count.item() / batch_size / epoch
    return acc_rate


if __name__ == "__main__":
    train()
    # print(test())
