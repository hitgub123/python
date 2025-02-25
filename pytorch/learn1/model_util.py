import torch, os


def get_model_name(base_file_name,suffix=None):
    if suffix:
        model_name = os.path.basename(base_file_name).replace(".py", f"_{suffix}.pkl")
    else:
        model_name = os.path.basename(base_file_name).replace(".py", ".pkl")
    model_name = f"pytorch/learn1/models/{model_name}"
    return model_name


def save_model(model, model_name, V):
    torch.save({"V": V, "W": model.state_dict()}, model_name)


def load_model(model, model_name):
    V = 0
    if os.path.exists(model_name):
        model_params = torch.load(model_name)
        model.load_state_dict(model_params["W"])
        V = model_params["V"]
    return V
