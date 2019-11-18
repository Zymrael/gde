import torch

class PerformanceContainer(object):
    """ Simple data class for metrics logging."""
    def __init__(self, data:dict):
        self.data = data
        
    @staticmethod
    def deep_update(x, y):
        for key in y.keys():
            x.update({key: list(x[key] + y[key])})
        return x
    
def accuracy(y_hat, y):
    preds = torch.max(y_hat, 1)[1]
    return torch.mean((y == preds).float())