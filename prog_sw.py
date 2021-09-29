import os
import click
import glob
from numpy.lib.polynomial import RankWarning
import torch
import imageio
import PIL.Image
import pandas as pd
import numpy as np
import pickle5 as pickle
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from skimage import exposure
from skimage.transform import resize
from sklearn.preprocessing import normalize
import dnnlib
import legacy

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def func(elem):
    return int(elem.split('/')[-2].split('_')[-2])

def pred_w(G, wdict='wdict',
            test_folder='../OAI_Xray/OAIMOST_test/nonprogressive/*/projected_w.npz',
            if_comp = True):
    if if_comp:
        z_dic = load_obj(wdict)
        df = pd.read_csv('../OAI_distinguish/label/oai_ptest.csv')
        df = df[df['fast prog'] == 1.0]

        for z in tqdm(df['name baseline']):
            key = z.split('.')[0]
            pat = key.split('_')[0]
            w = np.load(os.path.join('./project', key, 'projected_w.npz'))['w']
            ws = np.array(list(z_dic.values()))[:,0,0,:]
            difs = normalize(ws, norm='l2') - normalize(w[:,0,:], norm='l2')
            l2 = np.sqrt(np.sum(difs**2, axis=1))
            df = pd.DataFrame(list(z_dic.keys()), columns=['name'])
            df['l2'] = l2
            df = df.sort_values('l2').reset_index(drop=True)
            counter = 0
            vdiffs = np.zeros_like(w)
            pats = []
            pats.append(pat)
            for nn in df['name'].tolist():
                pat_ = nn.split('_')[0]
                lr = nn[-1]
                if not pat_ in pats:
                    nns = sorted(glob.glob('./project/'+pat_+'_*_'+lr+'/projected_w.npz'), key=func)
                    nn0 = np.load(nns[0])['w']
                    nn1 = np.load(nns[-1])['w']
                    vdiffs += nn1 - nn0
                    counter += 1
                    pats.append(pat_)
                if counter > 0:
                    break
            
            vdiffs = vdiffs/counter
            for n in range(6):
                w_ = vdiffs*0.2*n + w
                ws = torch.from_numpy(w_).cuda()
                synth_image = G.synthesis(ws, noise_mode='const')
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                outdir = os.path.join('../OAI_Xray/smooth', key)
                Path(outdir).mkdir(parents=True, exist_ok=True)
                PIL.Image.fromarray(np.squeeze(synth_image), 'L').save(os.path.join(outdir, str(n)+'_'+z))


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
def run_pred(network_pkl: str,
    seed: int,):  
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    pred_w(G, if_comp=True)
    
if __name__ == "__main__":
    run_pred()
            
            
            


