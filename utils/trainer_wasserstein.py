import torch
import os
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.Distribution import MNIST, MnistGenerator, L2Ball, Dirac
from torch.utils import tensorboard
from architectures.simple_fc import SimpleFC
from utils import metrics, utilities, spline_utils
from utils.aggmo import AggMo
import matplotlib.pyplot as plt

class TrainerWasserstein:
    """
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Prepare dataset classes
        train_dataset = MNIST(config['training_options']['train_dataset_file'])
        val_dataset = MNIST(config['training_options']['val_dataset_file'])
        self.mnist_generator = MnistGenerator(device=self.device)

        print('Preparing the dataloaders')
        # Prepare dataloaders 
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["training_options"]["batch_size"], shuffle=True, num_workers=config["training_options"]["num_workers"], drop_last=True)
        self.batch_size = config["training_options"]["batch_size"]
        self.val_dataloader = DataLoader(val_dataset, batch_size=60, shuffle=False, num_workers=1)

        print('Building the model')
        # Build the model

        self.model = SimpleFC(config['net_params'], **config['activation_fn_params'])
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

        self.total_training_step = 0

       
    def set_optimization(self):
        """Initialize the optmizer"""

        optim_name = self.config["optimizer"]["type"]
        if optim_name == 'Adam':
            optimizer_type = torch.optim.Adam
        elif optim_name == 'SGD':
            optimizer_type = torch.optim.SGD
        elif optim_name == 'RMSprop':
            optimizer_type = torch.optim.RMSprop
        elif optim_name == 'AggMo':
            optimizer_type = AggMo
        else:
            raise ValueError('Need to provide a valid optimizer type')

        params_list = [{'params': spline_utils.get_no_spline_coefficients(self.model), 'lr': self.config["optimizer"]["lr_weights"]}]
        if self.model.using_splines:
            params_list.append({'params': spline_utils.get_spline_coefficients(self.model), \
                                'lr': self.config["optimizer"]["lr_spline_coeffs"]})

            if self.config["activation_fn_params"]["spline_scaling_coeff"]:
                params_list.append({'params': spline_utils.get_spline_scaling_coeffs(self.model), \
                                    'lr': self.config["optimizer"]["lr_spline_scaling_coeffs"]})

        self.optimizer = optimizer_type(params_list)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config['scheduler_gamma'])
        

    def train(self):
        
        best_val = 0.0
        self.model.train()
        for epoch in range(self.epochs+1):
            self.train_epoch(epoch)
            if epoch % 5 == 0:
                val_result = self.valid_epoch(epoch)     
                
            # SAVE CHECKPOINT
            if val_result > best_val and epoch % 5 == 0:
                best_val = val_result
                self.save_checkpoint(epoch)
        
        self.writer.flush()
        self.writer.close()
        

    def train_epoch(self, epoch):
        """
        """
        self.model.train()

        log = {}
        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        for batch_idx, data in enumerate(tbar):
            data_p1 = data[0].to(self.device).reshape(-1, 784)
            data_p2 = self.mnist_generator(self.batch_size).reshape(-1, 784)
            # data_p1 = self.dirac(self.batch_size)
            # data_p2 = self.l2_ball(self.batch_size)
            self.optimizer.zero_grad()

            wassserstein_loss = -1 * (torch.mean(self.model(data_p1)) - torch.mean(self.model(data_p2)))
                
            # regularization
            regularization = torch.zeros_like(wassserstein_loss)
            if self.model.using_splines and self.config['training_options']['lmbda'] > 0:
                regularization = self.config['training_options']['lmbda'] * self.model.TV2()

            total_loss = wassserstein_loss + regularization
            total_loss.backward()
            self.optimizer.step()
                
            log['train_loss'] = total_loss.detach().cpu().item()

            #if self.total_training_step % (10 * 128 // self.batch_size)  == 0:
            if self.config["activation_fn_params"]["spline_scaling_coeff"] & self.model.using_splines:
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
        mean_wasserstein_distance = 0.0
        tbar = tqdm(self.val_dataloader, ncols=135, position=0, leave=True)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar):
                data_p1 = data[0].to(self.device).reshape(-1, 784)
                data_p2 = self.mnist_generator(60).reshape(-1, 784)
                # data_p1 = self.dirac(60)
                # data_p2 = self.l2_ball(60)

                mean_wasserstein_distance += (torch.mean(self.model(data_p1)) - torch.mean(self.model(data_p2))) / 100

            # METRICS TO TENSORBOARD
            self.wrt_mode = 'val'
            self.writer.add_scalar(f'{self.wrt_mode}/Mean Wasserstein Distance', mean_wasserstein_distance, epoch)

            # Log parameters
            if epoch == self.epochs and self.model.using_splines:
                j = 1
                for module in self.model.modules_linearspline:
                    x = module.grid_tensor
                    y = module.lipschitz_coefficients
                    figures_list = []
                    for kk in range(x.shape[0]):
                        fig, ax = plt.subplots()
                        ax.grid()
                        ax.plot(x[kk,:].cpu().numpy(), y[kk,:].cpu().numpy())
                        figures_list.append(fig)
                        plt.close()
                    self.writer.add_figure(' Activation functions layer ' + str(j), figures_list, global_step=epoch)
                    j += 1
        
        return mean_wasserstein_distance


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