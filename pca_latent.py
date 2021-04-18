import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import os
import click
import imageio
from skimage.transform import resize
import glob
import torch
import PIL.Image
from pathlib import Path
import pickle5 as pickle
from natsort import natsorted
from sklearn.decomposition import PCA

import dnnlib
import legacy


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def pred_w(G, selected_n=10, n_layers=2, affect_nlayer=2, affect_all=False,
        wdict='wdict', 
        test_folder='./project_test/*/projected_w.npz'):
    z_dic = load_obj(wdict)
    ws = np.array(list(z_dic.values()))[:,0,0,:]
    pca = PCA()
    _ = pca.fit_transform(ws)
    eigen_vecs = pca.components_
    # cov = np.cov(np.array(list(z_dic.values()))[:,0,0,:].T)
    # print(cov.shape)
    # eigen_vals, eigen_vecs = np.linalg.eig(cov)
    x = np.zeros((512, 1))
    x[selected_n] = 1
    vec = np.matmul(eigen_vecs, x).T # shape is [1, 512]

    for z in glob.glob(test_folder):
        key = z.split('/')[-2]
        pat = key.split('_')[0]
        w = np.load(z)['w'] # shape is [1, 14, 512]
        delta_w = np.zeros_like(w)
        print(z)

        for n, alpha in enumerate(np.linspace(-30, 30, num=21)):
            if not affect_all:
                delta_w[:,n_layers:n_layers+affect_nlayer] += np.tile(alpha*vec[None, ...], (1, affect_nlayer, 1))
            else:
                delta_w += np.tile(alpha*vec[None, ...], (1, w.shape[1], 1))
            w_ = w + delta_w
            ws = torch.from_numpy(w_).cuda()
            synth_image = G.synthesis(ws, noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            outdir = os.path.dirname(z)
            name = str(selected_n)+'_'+str(n_layers)+'_'+str(affect_nlayer)
            outdir_ = os.path.join(outdir, 'pca', name)
            Path(outdir_).mkdir(parents=True, exist_ok=True)
            PIL.Image.fromarray(np.squeeze(synth_image), 'L').save(f'{outdir_}/pred_'+str(n)+'.png')


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
def run_pred(network_pkl: str,
    seed: int,):  
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    for basis in range(10):
        for n_layer in range(10):
            for n in range(1, 2):
                pred_w(G, basis, n_layer, n)

if __name__ == "__main__":
    # run_projection() # pylint: disable=no-value-for-parameter
    run_pred()
            
            
            


