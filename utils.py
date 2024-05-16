
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def count_params(model):
    """
    Count the parameters in a NN model: by layer and total number
    """
    out = {}
    for name, param in model.named_parameters():
        out[name] = sum(p.numel() for p in param.data)
    
    return out, sum(out.values())