import argparse
import json
import torch
from utils.trainer_denoiser import TrainerDenoiser
from utils.trainer_1d import Trainer1D
from utils.trainer_wasserstein import TrainerWasserstein
import os
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def main(args):
    
    # Set up directories for saving results

    if args.exp == 'denoising':
        config = json.load(open('configs/config_denoiser.json'))
        trainer_inst = TrainerDenoiser(config, args.device)
    elif args.exp == '1d':
        config = json.load(open('configs/config_1d.json'))
        trainer_inst = Trainer1D(config, seed, args.device)
    elif args.exp == 'wasserstein':
        config = json.load(open('configs/config_wasserstein.json'))
        trainer_inst = TrainerWasserstein(config, args.device)
    else:
        raise ValueError('Need to provide a valid exp name')

    
    exp_dir = os.path.join(config['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 

    trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-e', '--exp', default='denoising', type=str, help='Choose the type of experiment')
    parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
    args = parser.parse_args()

    main(args)