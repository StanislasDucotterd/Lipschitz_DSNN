import torch
import os
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.Distribution import MNIST, MnistGenerator
from torch.utils import tensorboard
from architectures.inv_resnet import InvResNet
from utils import metrics, utilities, spline_utils
import torch.autograd as autograd
from utils.mmd import MMD_multiscale
import matplotlib.pyplot as plt

class TrainerInvertible:
    """
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

        print('Building the model')
        # Build the model

        self.model = InvResNet(config['net_params'], **config['activation_fn_params'])
        self.model = self.model.to(device)
        
        print("Number of parameters in the model: ", self.model.get_num_params())

        print(self.model)
        
        # Set up the optimizer
        self.set_optimization()
        self.batch_size = config['training_options']['batch_size']
        self.epochs = config["training_options"]['epochs']

        # CHECKPOINTS & TENSOBOARD
        run_name = config['exp_name']
        self.checkpoint_dir = os.path.join(config['log_dir'], config["exp_name"], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        config_save_path = os.path.join(config['log_dir'], config["exp_name"], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)
        self.criterion = MMD_multiscale

        self.total_training_step = 0

       
    def set_optimization(self):
        """Initialize the optmizer"""


        params_list = [{'params': spline_utils.get_no_spline_coefficients(self.model), 'lr': self.config["optimizer"]["lr_weights"]}]
        if self.model.using_splines:
            params_list.append({'params': spline_utils.get_spline_coefficients(self.model), \
                                'lr': self.config["optimizer"]["lr_spline_coeffs"]})

            params_list.append({'params': spline_utils.get_spline_scaling_coeffs(self.model), \
                                'lr': self.config["optimizer"]["lr_spline_scaling_coeffs"]})

        self.optimizer = torch.optim.Adam(params_list)
        

    def train(self):
        
        best_val = 0.0
        self.model.train()
        for epoch in range(self.epochs+1):
            self.train_epoch(epoch)
            self.valid_epoch(epoch)             
        self.save_checkpoint(epoch)
        
        self.writer.flush()
        self.writer.close()
        

    def train_epoch(self, epoch):
        """
        """
        self.model.train()

        log = {}
        tbar = tqdm(range(100), ncols=135, position=0, leave=True)
        for _ in tbar:
            data = 5 * torch.bernoulli(torch.ones(self.batch_size, 1, device=self.device) * 0.5)
            data += torch.randn(data.shape, device=self.device)
            self.optimizer.zero_grad()

            z1 = torch.randn(self.batch_size, 1, device=self.device)
            z2 = torch.randn(self.batch_size, 1, device=self.device)
            pred = self.model(data)
            loss = self.criterion(pred, z1) + self.criterion(self.model.inverse(z2), data)
                
            # regularization
            regularization = torch.zeros_like(loss)
            if self.model.using_splines and self.config['activation_fn_params']['lmbda'] > 0:
                regularization = self.config['activation_fn_params']['lmbda'] * self.model.TV2()

            total_loss = loss + regularization
            total_loss.backward()
            self.optimizer.step()
                
            log['train_loss'] = total_loss.detach().cpu().item()

            if self.model.using_splines:
                spline_scaling_coeffs = torch.nn.utils.parameters_to_vector(spline_utils.get_spline_scaling_coeffs(self.model))
                log['spline_scaling_coeff_mean'] = torch.mean(spline_scaling_coeffs).cpu().item()
                log['spline_scaling_coeff_std'] = torch.std(spline_scaling_coeffs).cpu().item()

                # the step actually represents the amount of data that has been in the model
            self.wrt_step = self.total_training_step * self.batch_size
            self.write_scalars_tb(log)

            tbar.set_description('T ({}) | TotalLoss {:.5f} |'.format(epoch, log['train_loss'])) 
            self.total_training_step += 1

        return log

    def valid_epoch(self, epoch):
        
        self.model.eval()
        
        with torch.no_grad():
            # METRICS TO TENSORBOARD
            self.wrt_mode = 'val'

            data = torch.randn(100000, 1, device=self.device)
            pred = self.model.inverse(data)
            fig, ax = plt.subplots()
            ax.hist(pred.cpu().numpy(), bins=1000)
            self.writer.add_figure('Prediction', fig, global_step=epoch)

            # Log parameters
            # if epoch == self.epochs and self.model.using_splines:
            #     j = 1
            #     for module in self.model.modules_linearspline:
            #         x = module.grid_tensor
            #         y = module.lipschitz_coefficients
            #         figures_list = []
            #         for kk in range(x.shape[0]):
            #             if kk % 16 == 0:
            #                 fig, ax = plt.subplots()
            #                 ax.grid()
            #             ax.plot(x[kk,:].cpu().numpy(), y[kk,:].cpu().numpy())
            #             if kk % 16 == 0:
            #                 figures_list.append(fig)
            #                 plt.close()
            #         self.writer.add_figure(' Activation functions layer ' + str(j), figures_list, global_step=epoch)
            #         j += 1


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint_best_epoch.pth'
        torch.save(state, filename)