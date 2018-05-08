
def num_of_parameters(model):
    params = 0
    for p in model.parameters():
        params += p.numel()
    return params

def get_state_dict(model):
    model_is_cuda = next(model.parameters()).is_cuda
    model = model.module if model_is_cuda else model
    return model.state_dict()