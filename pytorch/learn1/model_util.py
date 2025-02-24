import torch, os


def get_model_name(suffix):
    if suffix:
        model_name = os.path.basename(__file__).replace(".py", "_{}.pkl".format(suffix))
    else:
        model_name = os.path.basename(__file__).replace(".py", ".pkl")
    model_name = "pytorch/learn1/models/{}".format(model_name)
    return model_name


def save_model(model, model_name, max_rate):
    torch.save({"V": max_rate, "W": model.state_dict()}, model_name)


def load_model(model, model_name):
    V = 0
    if os.path.exists(model_name):
        model_params = torch.load(model_name)
        model.load_state_dict(model_params["W"])
        V = model_params["V"]
    return V
