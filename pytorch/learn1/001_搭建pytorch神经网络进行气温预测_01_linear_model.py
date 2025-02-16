import torch as tc,numpy as np,pandas as pd,datetime
from sklearn import preprocessing

# import os
# print(os.getcwd())

features=pd.read_csv('pytorch/learn1/datas/temps.csv')
# 独热编码
features=pd.get_dummies(features)
# print(features.head(),features.shape)

years=features['year']
months=features['month']
days=features['day']
dates=['{}-{}-{}'.format(y,m,d) for y,m,d in zip(years,months,days)]
dates=[datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]

labels=features['actual']
features=features.drop('actual',axis=1)
features=features.drop('friend',axis=1)
titles=list(features.columns)
# print(type(labels),type(features),features.head(),features.shape,titles)

features=np.array(features,dtype=np.float64)
features=preprocessing.StandardScaler().fit_transform(features)

inputs=tc.tensor(features,dtype=float)
labels=tc.tensor(np.array(labels).reshape(-1,1),dtype=float)
# 归一化到 [-1, 1]
# min_vals = inputs.min(dim=0, keepdim=True)[0]
# max_vals = inputs.max(dim=0, keepdim=True)[0]
# 标准化到 [0, 1]
# inputs = (inputs - min_vals) / (max_vals - min_vals).clamp(min=1e-6)
# print(inputs.size(),labels.shape,inputs)

#####################################################
#################### model ##########################
#####################################################
epochs=1000
learn_rate=0.01
model=tc.nn.Linear(inputs.shape[-1],labels.shape[-1],dtype=tc.float64)
optimizer=tc.optim.SGD(model.parameters(),lr=learn_rate)
criterion=tc.nn.MSELoss()

for i in range(epochs):
    optimizer.zero_grad()
    outputs=model(inputs)
    loss=criterion(labels,outputs)
    loss.backward()
    optimizer.step()
    if not i%100:
        print(i,loss.item())


