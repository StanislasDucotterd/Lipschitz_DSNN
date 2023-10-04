import argparse
import json
import torch
from utils.trainer_denoiser import TrainerDenoiser
from utils.trainer_1d import Trainer1D
from utils.trainer_wasserstein import TrainerWasserstein
from utils.trainer_wgan import TrainerWGAN
from utils.trainer_invertible import TrainerInvertible
import os
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def main(args):

    config = json.load(open(args.config))

    if args.exp == 'denoising':
        trainer_inst = TrainerDenoiser(config, args.device)
    elif args.exp == '1d':
        trainer_inst = Trainer1D(config, config['seed'], args.device)
    elif args.exp == 'wasserstein':
        trainer_inst = TrainerWasserstein(config, args.device)
    elif args.exp == 'wgan':
        trainer_inst = TrainerWGAN(config, args.device)
    elif args.exp == 'invertible':
        trainer_inst = TrainerInvertible(config, args.device)
    else:
        raise ValueError('Need to provide a valid exp name')
    
    exp_dir = os.path.join(config['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed']) 
    torch.set_num_threads(1)

    trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_invertible.json', type=str, help='Choose config file')
    parser.add_argument('-e', '--exp', default='invertible', type=str, help='Choose the type of experiment')
    parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
    args = parser.parse_args()

    main(args)