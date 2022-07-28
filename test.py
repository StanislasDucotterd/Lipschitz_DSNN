import torch
import numpy as np
from tqdm import tqdm
import scipy.stats
from torch.utils.data import DataLoader
from dataloader.BSD500 import BSD500
import matplotlib.pyplot as plt
from architectures.simple_cnn import SimpleCNN
from utils import metrics, utilities, spline_utils

device = 'cuda:0'
val_dataset = BSD500("/home/ducotter/PnP/data/test.h5")
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
tbar_val = tqdm(val_dataloader, ncols=130, position=0, leave=True)

EXP_NAME1 = 'sigma_5/DS_layer_5_channel_64_region_50_spectral_norm_sigma_5/'
EXP_NAME2 = 'sigma_5/HH_layer_5_channel_64_spectral_norm_sigma_5/'

infos1 = torch.load('denoising_exps/' + EXP_NAME1 + 'checkpoints/checkpoint_best_epoch.pth')
config1 = infos1['config']
infos2 = torch.load('denoising_exps/' + EXP_NAME2 + 'checkpoints/checkpoint_best_epoch.pth')
config2 = infos2['config']

model1 = SimpleCNN(config1['net_params'], **config1['activation_fn_params']).to(device)
model1.load_state_dict(infos1['state_dict'])
model1.eval()

model2 = SimpleCNN(config2['net_params'], **config2['activation_fn_params']).to(device)
model2.load_state_dict(infos2['state_dict'])
model2.eval()

sigma = 5
psnr_val_diffs = []
ssim_val_diffs = []

for batch_idx, data in enumerate(tbar_val):
    data = data.to(device)
    noisy_data = data + (sigma/255.0)*torch.randn(data.shape, device=device)

    output1 = (noisy_data + model1(noisy_data))/2.0
    output2 = (noisy_data + model2(noisy_data))/2.0

    out_val1 = torch.clamp(output1, 0., 1.)
    out_val2 = torch.clamp(output2, 0., 1.)

    psnr_val_diffs.append(utilities.batch_PSNR(out_val1, data, 1.)-utilities.batch_PSNR(out_val2, data, 1.))
    ssim_val_diffs.append(utilities.batch_SSIM(out_val1, data, 1.)-utilities.batch_SSIM(out_val2, data, 1.))

fig, ax = plt.subplots()
ax.grid()
ax.hist(psnr_val_diffs, bins=[-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
fig.savefig('hist_psnr.png')
plt.close()

fig, ax = plt.subplots()
ax.grid()
ax.hist(ssim_val_diffs, bins=[-0.003, 0.000, 0.003, 0.006, 0.009, 0.012, 0.015])
ax.set_xticks([-0.003, 0.000, 0.003, 0.006, 0.009, 0.012, 0.015])
fig.savefig('hist_ssim.png')
plt.close()

print(scipy.stats.wilcoxon(psnr_val_diffs))
print(scipy.stats.wilcoxon(ssim_val_diffs))
print(scipy.stats.ttest_1samp(psnr_val_diffs, 0))
print(scipy.stats.ttest_1samp(ssim_val_diffs, 0))