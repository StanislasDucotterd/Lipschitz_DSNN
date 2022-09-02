import numpy as np
import torch
from torch.utils.data import Dataset

def ReLU(x):
    if x >= 0: return x
    else: return 0

def slope_1_ae(x, n, seed):
    "define random function with slope 1 or -1 almost everywhere"
    "and changes of slope uniformly at random n times between -1 and 1"
    np.random.seed(seed)
    knots = sorted(2.0 * np.random.rand(n) - 1.0)
    value = ReLU(x + 1.0)
    mean_value = 2
    for i, knot in enumerate(knots):
        value += ReLU(x - knot) * 2 * (-1)**(i+1) 
        mean_value += (1 - 2*knot + knot**2) * (-1)**(i+1)
    return value - mean_value / 2

def slope_1_flat(x, n, seed):
    "define random function with slope 1 or -1 or 0 almost everywhere"
    "and changes of slope uniformly at random n times between -1 and 1"
    np.random.seed(seed)
    knots = np.random.rand(2*n)
    knots = sorted(2.0 * knots[n:2*n] - 1.0)
    value = ReLU(x + 1.0)
    current_state = 1
    mean_value = 2
    for knot in knots:
        slope_change = np.random.randint(2)
        if current_state == 1: slope_change = -slope_change - 1
        elif current_state == 0: slope_change = 2 * slope_change - 1
        else: slope_change = slope_change + 1 
        value += ReLU(x - knot) * slope_change 
        mean_value += slope_change * (0.5 - knot + 0.5*knot**2)
        current_state = current_state + slope_change
    return value - mean_value / 2

def cosines(x):
    return np.cos(7*np.pi*x)/(7*np.pi)

def sawtooth(x):
    triangle = lambda x: 1 - np.abs(x) if np.abs(x) <= 1 else 0
    sawtooth = lambda x: (triangle(4*(x + 0.75)) + triangle(4*(x + 0.25)) + \
                triangle(4*(x - 0.25)) + triangle(4*(x - 0.75)))/4
    
    return sawtooth(x) - 0.125 

def threshold(x):
    tau = 1/20
    if x <= -tau:
        return x + 2*tau
    elif x <= tau:
        return -x
    else:
        return x - 2*tau 

def generate_testing_set(f, n_points):
    X = np.linspace(-1.0, 1.0, n_points).astype(np.float32)
    y = np.vectorize(f, otypes=[np.float32])(X)
    X, y = torch.tensor(X), torch.tensor(y)
    X, y = X.unsqueeze(1), y.unsqueeze(1)
    
    return X, y

class Function1D(Dataset):

    def __init__(self, function, n_points, nbr_models, seed):
        
        np.random.seed(seed)
        X = (2.0 * np.random.rand(n_points, nbr_models, 1) - 1.0).astype(np.float32)
        y = np.vectorize(function)(X).astype(np.float32)
        self.X, self.y = torch.tensor(X), torch.tensor(y)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:,:], self.y[idx,:,:]