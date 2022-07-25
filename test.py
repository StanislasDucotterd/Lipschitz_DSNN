import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.BSD500 import BSD500
from architectures.simple_cnn import SimpleCNN
from utils import metrics, utilities, spline_utils

device = 'cuda:0'
val_dataset = BSD500("data/BSD500/test.h5")
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
tbar_val = tqdm(val_dataloader, ncols=130, position=0, leave=True)

EXP_NAME = 'sigma_10/GS_layer_5_channel_64_groupsize_4_spectral_norm_sigma_10/'
infos = torch.load('denoising_exps/' + EXP_NAME + 'checkpoints/checkpoint_best_epoch.pth')
config = infos['config']

model = SimpleCNN(config['net_params'], **config['activation_fn_params']).to(device)
model.load_state_dict(infos['state_dict'])
model.eval()

sigma = 10
psnr_val = []
ssim_val = []

for batch_idx, data in enumerate(tbar_val):
    data = data.to(device)
    noisy_data = data + (sigma/255.0)*torch.randn(data.shape, device=device)

    output = (noisy_data + model(noisy_data))/2.0

    out_val = torch.clamp(output, 0., 1.)
    psnr_val.append(utilities.batch_PSNR(out_val, data, 1.))
    ssim_val.append(utilities.batch_SSIM(out_val, data, 1.))

print('PSNR Mean: ' + str(np.mean(psnr_val)))
print('PSNR Std: ' + str(np.std(psnr_val)))
print('SSIM Mean: ' + str(np.mean(ssim_val)))
print('SSIM Std: ' + str(np.std(ssim_val)))