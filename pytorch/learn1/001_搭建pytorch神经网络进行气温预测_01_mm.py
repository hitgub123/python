import torch as tc,numpy as np,pandas as pd
from sklearn import preprocessing

features=pd.read_csv('pytorch/learn1/datas/temps.csv')
# 独热编码
features=pd.get_dummies(features)

labels=features['actual']
features=features.drop(['actual','friend'],axis=1)
features=np.array(features,dtype=np.float64)
features=preprocessing.StandardScaler().fit_transform(features)

inputs=tc.tensor(features,dtype=float)
labels=tc.tensor(np.array(labels).reshape(-1,1),dtype=float)

#####################################################
#################### model ##########################
#####################################################
hidden_size=128
W1=tc.rand((inputs.shape[-1],hidden_size),requires_grad=True,dtype=float)
B1=tc.rand(hidden_size,requires_grad=True,dtype=float)
W2=tc.rand(hidden_size,1,requires_grad=True,dtype=float)
B2=tc.rand(hidden_size,requires_grad=True,dtype=float)

epochs=10000
lr=0.001
for i in range(epochs):
    outputs=inputs.mm(W1)+B1
    outputs=tc.relu(outputs)
    predictions=outputs.mm(W2)+B2
    loss=tc.mean((predictions-labels)**2)
    loss.backward()

    if not i%100:
        print(i,loss)
    
    W1.data.add_(-lr*W1.grad.data)
    B1.data.add_(-lr*B1.grad.data)
    W2.data.add_(-lr*W2.grad.data)
    B2.data.add_(-lr*B2.grad.data)

    W1.grad.data.zero_()
    B1.grad.data.zero_()
    W2.grad.data.zero_()
    B2.grad.data.zero_()