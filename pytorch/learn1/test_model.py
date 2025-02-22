import torch as tc, numpy as np, pandas as pd,os
from sklearn import preprocessing
from sklearn.datasets import fetch_openml


model_name = os.path.basename(__file__).replace(".py", ".pkl")
model_name = "pytorch/learn1/models/{}".format(model_name)
print(model_name)
