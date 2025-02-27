import torch as tc, numpy as np, pandas as pd
from sklearn import preprocessing

features = pd.read_csv("pytorch/learn1/datas/temps.csv")
# 独热编码
features = pd.get_dummies(features)

labels = features["actual"]
features = features.drop(["actual", "friend"], axis=1)
features = np.array(features, dtype=np.float64)
features = preprocessing.StandardScaler().fit_transform(features)

inputs = tc.tensor(features, dtype=float)
labels = tc.tensor(np.array(labels).reshape(-1, 1), dtype=float)

#####################################################
#################### model ##########################
#####################################################
hidden_size = 128
data_size = inputs.shape[0]
input_size = inputs.shape[-1]
output_size = 1
batch_size = 16
lr = 0.001
epochs = 1000

model = tc.nn.Sequential(
    tc.nn.Linear(input_size, hidden_size, dtype=float),
    # tc.nn.ReLU(),
    tc.nn.Sigmoid(),
    tc.nn.Linear(hidden_size, output_size, dtype=float),
)
cost = tc.nn.MSELoss(reduction="mean")
optimizer = tc.optim.Adam(model.parameters(), lr=lr)

for i in range(epochs):
    batch_loss = []
    for start in range(0, data_size, batch_size):
        optimizer.zero_grad()
        end = start + batch_size if start + batch_size < data_size else data_size
        x, y = inputs[start:end], labels[start:end]
        prediction = model(x)
        loss = cost(prediction, y)
        loss.backward()
        # loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.item())

    if not i % 100:
        print(i, tc.tensor(batch_loss).mean().item())
model_name = "pytorch/learn1/models/搭建pytorch神经网络进行气温预测.pkl"
tc.save(model.state_dict(), model_name)
# model.load_state_dict(tc.load(model_name,weights_only=0))
