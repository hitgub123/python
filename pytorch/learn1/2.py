import torch as tc,numpy as np

x=np.array([i for i in range(11)],np.float32)
y=x*2+1
x_train=x.reshape(-1,1)
y_train=y.reshape(-1,1)
# print(x_train,y_train)
inputs=tc.from_numpy(x_train)

model=tc.nn.Linear(1,1)
p=model(inputs.requires_grad_())
print(p)

model.load_state_dict(tc.load("2x_1.pkl",weights_only=0))
p=model(inputs.requires_grad_())
print(p)