import torch as tc,numpy as np

device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
print(device)

x=np.array([i for i in range(11)],np.float32)
y=x*2+1
x_train=x.reshape(-1,1)
y_train=y.reshape(-1,1)
# print(x_train,y_train)

model=tc.nn.Linear(1,1)
# print(model)

epochs=1000
learn_rate=0.01
optimizer=tc.optim.SGD(model.parameters(),lr=learn_rate)
criterion=tc.nn.MSELoss()

inputs=tc.from_numpy(x_train).to(device)
labels=tc.from_numpy(y_train).to(device)

for i in range(epochs):
    optimizer.zero_grad()
    outputs=model(inputs)
    loss=criterion(labels,outputs)
    loss.backward()
    optimizer.step()
    # if not i%50:
    #     print(i,loss.item())
# x=np.array([i*3 for i in range(10)],np.float32)
# x=x.reshape(-1,1)
# print(model(tc.from_numpy(x)))
# print(x*2+1)
tc.save(model.state_dict(),"2x_1.pkl")
p=model(inputs.requires_grad_())
print(p)




