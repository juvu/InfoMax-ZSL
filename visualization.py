import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import numpy as np
import classifier
import classifier2
import sys
import traceback
import model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors.kde import KernelDensity
from tensorboardX import SummaryWriter
from main import get_args, generate_syn_feature
import matplotlib.pyplot as plt
import pickle
from tqdm import trange


def dimension_reduction(feature, n=2, method='pca'):
    if method == 'pca':
        pca = PCA(n_components=n, svd_solver='full')
        new_feature = pca.fit_transform(feature)
    elif method == 'tsne':
        new_feature = TSNE(n_components=n).fit_transform(feature)
    elif method == 'first':
        new_feature = feature[:, :2]
    return new_feature

def get_real_feature(data):
    return data.test_unseen_feature, data.test_unseen_label

def reset_seed(opt):
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

def load_G(filename, opt):
    opt.netG = filename
    netG = model.MLP_G(opt)
    netG.load_state_dict(torch.load(os.path.join(f'{opt.outf}{opt.outname}', opt.netG)))
    netG = netG.cuda()
    return netG


def gen_data():
    X = {}
    y = {}
    print('generating feature...')
    for k, G in netG.items():
        if G is None: # Real
            X[k], y[k] = get_real_feature(data)
        else:
            reset_seed(opt)
            X[k], y[k] = generate_syn_feature(G, data.unseenclasses, data.attribute, opt.syn_num, opt)
        y[k] = util.map_label(y[k], data.unseenclasses)
        X[k] = X[k].cpu().numpy()
        y[k] = y[k].cpu().numpy()

    print('done')
    return X, y

def visual_2d(samples, label=0, comment=''):
    plt.figure()
    reset_seed(opt)
    for k, G in netG.items():
        X, y = samples[0][k], samples[1][k]
        if label != 'all':
            X = X[y == label]
            y = y[y == label]
        if opt.real:
            num = len(y)
        else:
            pass
            # X = X[:num]
            # y = y[:num]
        if X.shape[1] > 2:
            print('performing dimension reduction...')
            X_ = dimension_reduction(X)
        else:
            X_ = X
        if G is None:
            X_ = X_[:200]
        plt.scatter(X_[:, 0], X_[:, 1])
        print(k, f'number of samples: {len(X_)}')
        plt.savefig(f'diagram/{k}.jpg')
    plt.legend(list(netG.keys()), loc='upper right')
    plt.savefig(f'diagram/visual_2d_{opt.outname}_{label}_{comment}.jpg')
    plt.savefig(f'diagram/visual_2d_{opt.outname}_{label}_{comment}.pdf')

def visual_kde(samples, comment=''):
    N = len(data.unseenclasses)
    scores = {}
    for k, G in netG.items():
        if G is not None:
            scores[k] = np.zeros(N)
    import time
    for i in range(N):
        start = time.time()
        print(i)
        X_real, y_real = samples[0]['Real'], samples[1]['Real']
        cond = y_real == i
        X_real, y_real = X_real[cond], y_real[cond]
        kde = KernelDensity().fit(dimension_reduction(X_real))
        for k, G in netG.items():
            if G is not None:
                X, y = samples[0][k], samples[1][k]
                X = X[y == i]
                scores[k][i] = np.exp(kde.score_samples(dimension_reduction(X)).mean())
                print(k, scores[k][i])
        end = time.time()
        print(f'time elapsed: {end - start}')

    plt.figure()
    classes = np.arange(N)
    for k, G in netG.items():
        if G is not None:
            plt.plot(classes, scores[k])
    print(np.argsort(scores['GAN + MI'] - scores['GAN']))
    plt.legend(list(netG.keys())[:-1])
    plt.savefig(f'diagram/visual_kde{comment}.jpg')
    plt.savefig(f'diagram/visual_kde{comment}.pdf')

def get_reduced_data(samples, n=2, method='tsne', per_class=True, **kwarg):
    N = len(data.unseenclasses)
    if 'label' in kwarg.keys():
        label = kwarg['label']
        print(f'for label {label} only')
        L = np.sum([np.sum(samples[1][k] == label) for k in netG.keys()])
        X_ = np.zeros((0, n))
        y_ = np.zeros((0))
        print(X_.shape)
        idx = 0 
        indices = []
        for k, G in netG.items():
            X, y = samples[0][k], samples[1][k]
            X = X[y == label]
            y = y[y == label]
            indices.append((idx, idx + len(y)))
            X_ = np.concatenate((X_, dimension_reduction(X, n, method)))
            y_ = np.concatenate((y_, y))
            print(X_.shape)
            idx += len(y)
        print(indices)
        record = {
                'X':    X_,
                'y':    y_,
                'idx':  indices
        }
        pickle.dump(record, open(f'reduced/{method}_{label}_{opt.syn_num}.p', 'wb'))
    else:
        for k, G in netG.items():
            X, y = samples[0][k], samples[1][k]
            X_ = np.zeros((len(X), n))
            print(f'X: {X.shape}')
            if per_class:
                for i in trange(N):
                    print(k, i)
                    X_i = X[y == i]
                    X_[y == i] = dimension_reduction(X_i, n, method)
            else:
                X_ = dimension_reduction(X, n, method)
            pickle.dump(X_, open(f'reduced/{opt.outname}_{method}_{n}_{k}.p', 'wb'))
            pickle.dump(y, open(f'reduced/{opt.outname}_{method}_{n}_{k}_y.p', 'wb'))

def load_reduced_data(n=2, method='tsne', per_class=True, **kwarg):
    X = {}
    y = {}
    if 'label' in kwarg: # the reduced data is from all three kinds of source(Real, GAN, GAN + MI) with class `label'
        label = kwarg['label']
        record = pickle.load(open(f'reduced/{method}_{label}_{opt.syn_num}.p', 'rb'))
        idx = record['idx']
        X_ = record['X']
        y_ = record['y']
        for i, k in enumerate(netG.keys()):
            X[k] = X_[idx[i][0]: idx[i][1]]
            y[k] = y_[idx[i][0]: idx[i][1]]
            print(len(X[k]), len(y[k]))
        return X, y

    for k, G in netG.items():
        X[k] = pickle.load(open(f'reduced/{opt.outname}_{method}_{n}_{k}.p', 'rb'))
        y[k] = pickle.load(open(f'reduced/{opt.outname}_{method}_{n}_{k}_y.p', 'rb'))
        # X[k] = pickle.load(open(f'reduced/{method}_{n}_{k}_{per_class}.p', 'rb'))
        # y[k] = pickle.load(open(f'reduced/{method}_{n}_{k}_y_{per_class}.p', 'rb'))
    return X, y

def init_nets():
    netG = {}
    netG['GAN'] = load_G('netG_80.pth', opt)
    netG['GAN + MI'] = load_G('netG_80_KL.pth', opt)
    netG['Real'] = None
    # netG['GAN + vMI'] = load_G('netG_199_KL_visual_mi.pth', opt)
    return netG

if __name__ == '__main__':
    parser = get_args()
    parser.add_argument('--real', action='store_true', default=False, help='whether to use real')
    opt = parser.parse_args()
    print('visualization')
    opt = parser.parse_args()
    print(opt)

    data = util.DATA_LOADER(opt)
    netG = init_nets()

    print(data.unseenclasses)
    load = True
    if load:
        X, y = load_reduced_data(method='pca')
        visual_kde((X, y), comment=f'{opt.outname}_pca')
        for i in range(data.unseenclasses.size(0)):
            visual_2d((X, y), label=i, comment='pca')
    else:
        X, y = gen_data()
        get_reduced_data((X, y), method='pca')

    # visual_2d((X, y), label='all', comment='tsne')
    # visual_2d((X, y), label=49, comment='tsne')
