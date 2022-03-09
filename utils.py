

import os
import pickle

import numpy as np
import torch

from scipy import stats
from scipy.signal import lfilter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_bins(data):
    # Freedman-Diaconis rule
    data = np.asarray(data)
    n = data.size
    v25, v75 = np.percentile(data, [25, 75])
    dx = 2 * (v75 - v25) / (n ** (1 / 3))
    n_bins = int((data.max() - data.min())/dx)
    return n_bins

def np_describe(arr):
    mean, sd = np.mean(arr), np.std(arr)
    # min, max, median, range, variance = np.amin(arr), np.amax(arr), np.median(arr), np.ptp(arr), np.var(arr) 
    print(f'Mean ={mean:.3f}',
            f'Standard Deviation ={sd:.3f}')
    return mean, sd



def save_pickle(name, dict_to_save, force=False):
    if force or not os.path.exists(f'{name}.pickle') or os.stat(f'{name}.pickle').st_size==0:
        print("saving pickle")
        with open(f'{name}.pickle', 'wb') as handle:
            pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    if os.path.exists(f'{name}.pickle'):
        print('loading pickle')
        with open(f'{name}.pickle', 'rb') as handle:
            return pickle.load(handle)
    else:
        print("File Not Found")


def calc_p(gen_samples, alpha = 1e-3):
    test_size = len(gen_samples)
    data_fn = lambda x: torch.randn((x, 1), device='cpu')
    a = data_fn(test_size).cpu().numpy().flatten()
    b = gen_samples
    x = np.concatenate((a, b))
    k2, p = stats.normaltest(x)
    
    # print("p = {:g}".format(p))
    # if p < alpha:  # null hypothesis: x comes from a normal distribution
    #     print("rejected")
    # else:
    #     print("not rejected")
    return k2

def calc_w(gen_samples):  
    shapiro_test = stats.shapiro(gen_samples)
    # p = shapiro_test.pvalue
    # if p <= 0.05:  # null hypothesis: x comes from a normal distribution
    #     print("rejected, so probably not normal")
    # else:
    #     print("not rejected, so probably normal")
    return shapiro_test.statistic

def ema(samples, alpha = 0.9):
     # alpha smoothing coefficient
    zi = [samples[0]] # seed the filter state with first value
    # filter can process blocks of continuous data if <zi> is maintained
    y, zi = lfilter([1.-alpha], [1., -alpha], samples, zi=zi)
    return y