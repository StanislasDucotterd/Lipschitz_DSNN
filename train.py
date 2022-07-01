import argparse
import json
import torch
from utils.trainer_denoiser import TrainerDenoiser
import os
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def main(args):
    
    # Set up directories for saving results
    config = json.load(open(args.config))
    exp_dir = os.path.join(config['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args.config.endswith('denoiser.json'):
        trainer_inst = TrainerDenoiser(config, seed, args.device)
        trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_denoiser.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='device to use')
    args = parser.parse_args()

    main(args)