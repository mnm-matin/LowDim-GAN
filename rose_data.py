import numpy as np
import torch
from torch.distributions.uniform import Uniform


def target_function(Z):
    '''
    Map Z ([-1,1], [-1,1]) to a rose figure
    '''
    X = Z[:, 0]
    Y = Z[:, 1]
    theta = X * np.pi
    r = 0.05*(Y+1) + 0.90*abs(np.cos(2*theta))
    polar = np.zeros((Z.shape[0], 2))
    polar[:, 0] = r * np.cos(theta)
    polar[:, 1] = r * np.sin(theta)
    return polar

def sample_from_target_function(samples):
    '''
    sample from the target function
    '''
    generate_noise = lambda samples: np.random.uniform(-1, 1, (samples, 2))
    Z = generate_noise(samples)
    return torch.from_numpy(target_function(Z).astype('float32'))

def generate_noise(samples):
    '''
    Generate `samples` samples of uniform noise in 
    ([-1,1], [-1,1])
    '''
    return np.random.uniform(-1, 1, (samples, 2))

def sample_noise(samples):
    '''
    Generate `samples` samples of uniform noise in 
    ([-1,1], [-1,1])
    '''
    return Uniform(-1, 1).sample((samples,2)) 