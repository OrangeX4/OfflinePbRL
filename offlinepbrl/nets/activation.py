import torch

def get_activation(name):
    if name == "identity":
        return torch.nn.Identity
    elif name == "sigmoid":
        return torch.nn.Sigmoid
    elif name == "tanh":
        return torch.nn.Tanh
    elif name == "relu":
        return torch.nn.ReLU
    elif name == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise ValueError(f"Unknown reward activation: {name}")