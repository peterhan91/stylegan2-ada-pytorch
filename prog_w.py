import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import os
import click
from skimage.transform import resize
import glob
import torch
import PIL.Image
import imageio
import pickle5 as pickle
from pathlib import Path
from natsort import natsorted

import dnnlib
import legacy


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def func(elem):
    return int(elem.split('/')[-2].split('_')[-2])

def plot(test_folder):
    if test_folder.split('/')[-3].split('_')[0] == 'stable':
        surfix = 's'
    elif test_folder.split('/')[-3].split('_')[0] == 'progressive':
        surfix = 'p'

    outdir = './project_test_/result_m' + surfix
    Path(outdir).mkdir(parents=True, exist_ok=True)

    dir = './project_test_/' + test_folder.split('/')[-3]
    dir_ = './project_test_/' + test_folder.split('/')[-3] + '_'

    for target in glob.glob(os.path.join(dir, '*/target.png')):
        try:
            path = os.path.dirname(target)
            pat = target.split('/')[-2]
            lr = pat[-1]
            init = imageio.imread(target)
            reco = imageio.imread(os.path.join(path, 'proj.png'))
            pred = imageio.imread(os.path.join(path, 'pred.png'))
            pat_ = pat.split('_')[0] + '_*_' + lr + '/target.png'
            print(glob.glob(os.path.join(dir_, pat_)))
            late = imageio.imread(glob.glob(os.path.join(dir_, pat_))[0])
            upper = np.concatenate((init, reco), axis=1)
            lower = np.concatenate((late, pred), axis=1)
            total = np.concatenate((upper, lower), axis=0)
            imageio.imwrite(os.path.join(outdir, pat+'.png'), total)
        except IndexError:
            print(pat)
            pass

def pred_w(G,
        wdict='wdict', 
        test_folder='./project_test_/progressive_most/*/projected_w.npz',
        if_comp = True):
    if if_comp:
        z_dic = load_obj(wdict)
        dic_most = load_obj('wdict_most')
        z_dic.update(dic_most)
        print(len(z_dic))

        for z in glob.glob(test_folder):
            key = z.split('/')[-2]
            pat = key.split('_')[0]
            w = np.load(z)['w']
            ws = np.array(list(z_dic.values()))[:,0,0,:]
            # print('total dict length: ', len(ws))
            difs = normalize(ws, norm='l2') - normalize(w[:,0,:], norm='l2')
            l2 = np.sqrt(np.sum(difs**2, axis=1))
            # l2 = normalize(ws, norm='l2')
            df = pd.DataFrame(list(z_dic.keys()), columns=['name'])
            df['l2'] = l2
            df = df.sort_values('l2').reset_index(drop=True)

            counter = 0
            vdiffs = np.zeros_like(w)
            for nn in df['name'].tolist():
                pat_ = nn.split('_')[0]
                lr = nn[-1]
                if not pat == pat_:
                    nns = sorted(glob.glob('./project/'+pat_+'_*_'+lr+'/projected_w.npz'), key=func)
                    nns_ = sorted(glob.glob('./project_most/'+pat_+'_*_'+lr+'/projected_w.npz'), key=func)
                    if len(nns_) > 0:
                        nns = nns_
                        print('using MOST nn!', pat)
                    nn0 = np.load(nns[0])['w']
                    nn1 = np.load(nns[-1])['w']
                    # print(nns[0], nns[-1])
                    vdiffs += nn1 - nn0
                    counter += 1
                
                if counter > 0:
                    break
            vdiffs = vdiffs / counter

            w_ = vdiffs + w
            ws = torch.from_numpy(w_).cuda()
            synth_image = G.synthesis(ws, noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            outdir_ = os.path.dirname(z)
            PIL.Image.fromarray(np.squeeze(synth_image), 'L').save(f'{outdir_}/pred.png')
    plot(test_folder)


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

    pred_w(G, if_comp=False)
    

if __name__ == "__main__":
    # run_projection() # pylint: disable=no-value-for-parameter
    run_pred()
            
            
            


