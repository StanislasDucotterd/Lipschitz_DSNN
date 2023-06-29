import torch
import os
import json
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.Distribution import MNIST
from torch.utils import tensorboard
from architectures.simple_generator import SimpleGenerator
from architectures.simple_discriminator import SimpleDiscriminator
from utils import metrics, utilities, spline_utils
import matplotlib.pyplot as plt

class TrainerWGAN:
    """
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Prepare dataset classes
        train_dataset = MNIST(config['training_options']['train_dataset_file'], center=True)

        print('Preparing the dataloaders')
        # Prepare dataloaders 
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["training_options"]["batch_size"], shuffle=True, num_workers=config["training_options"]["num_workers"], drop_last=True)
        self.batch_size = config["training_options"]["batch_size"]
        self.val_dataset, _ = torch.load(config['training_options']['val_dataset_file'])
        self.val_dataset = self.val_dataset.unsqueeze(1)/127.5-1

        print('Building the model')
        # Build the model

        self.generator = SimpleGenerator(**config['generator_activation_fn_params'])
        self.generator = self.generator.to(device)

        self.discriminator = SimpleDiscriminator(**config['discriminator_activation_fn_params'])
        self.discriminator = self.discriminator.to(device)
        
        print("Number of parameters in the generator: ", self.generator.get_num_params())
        print("Number of parameters in the discriminator: ", self.discriminator.get_num_params())

        print(self.generator)
        print(self.discriminator)
        
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

        params_list = [{'params': spline_utils.get_no_spline_coefficients(self.generator), \
                        'lr': self.config["generator_optimizer"]["lr_weights"]}]
        if self.generator.using_splines:
            params_list.append({'params': spline_utils.get_spline_coefficients(self.generator), \
                                'lr': self.config["generator_optimizer"]["lr_spline_coeffs"]})

            params_list.append({'params': spline_utils.get_spline_scaling_coeffs(self.generator), \
                                'lr': self.config["generator_optimizer"]["lr_spline_scaling_coeffs"]})

        self.generator_optimizer = torch.optim.Adam(params_list, betas=(0.5, 0.999))

        params_list = [{'params': spline_utils.get_no_spline_coefficients(self.discriminator), \
                        'lr': self.config["discriminator_optimizer"]["lr_weights"]}]
        if self.discriminator.using_splines:
            params_list.append({'params': spline_utils.get_spline_coefficients(self.discriminator), \
                                'lr': self.config["discriminator_optimizer"]["lr_spline_coeffs"]})

            params_list.append({'params': spline_utils.get_spline_scaling_coeffs(self.discriminator), \
                                'lr': self.config["discriminator_optimizer"]["lr_spline_scaling_coeffs"]})

        self.discriminator_optimizer = torch.optim.Adam(params_list, betas=(0.5, 0.999))
        

    def train(self):
        
        for epoch in range(self.epochs+1):
            self.train_epoch(epoch)
            self.valid_epoch(epoch)     
        
        self.save_checkpoint(epoch)
        self.writer.flush()
        self.writer.close()
        

    def train_epoch(self, epoch):
        """
        """
        self.discriminator.train()
        self.generator.train()

        log = {}
        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        for batch_idx, data in enumerate(tbar):
            z = torch.rand(self.batch_size, 62, device=self.device)
            data_real = data[0].to(self.device)
            data_fake = self.generator(z)
            self.discriminator_optimizer.zero_grad()

            wassserstein_loss = torch.mean(self.discriminator(data_fake)) - torch.mean(self.discriminator(data_real))
                
            # regularization
            discriminator_regularization = torch.zeros_like(wassserstein_loss)
            if self.discriminator.using_splines and self.config['discriminator_activation_fn_params']['lmbda'] > 0:
                discriminator_regularization = self.config['discriminator_activation_fn_params']['lmbda'] * self.discriminator.TV2()

            discriminator_loss = wassserstein_loss + discriminator_regularization
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
                
            log['train_discriminator_loss'] = discriminator_loss.detach().cpu().item()

            if self.discriminator.using_splines:
                spline_scaling_coeffs = torch.nn.utils.parameters_to_vector(spline_utils.get_spline_scaling_coeffs(self.discriminator))
                log['discriminator_spline_scaling_coeff_mean'] = torch.mean(spline_scaling_coeffs).cpu().item()
                log['discriminator_spline_scaling_coeff_std'] = torch.std(spline_scaling_coeffs).cpu().item()

                # the step actually represents the amount of data that has been in the model

            tbar.set_description('T ({}) | DiscriminatorLoss {:.5f} |'.format(epoch, log['train_discriminator_loss'])) 

            if (batch_idx+1) % 5 == 0:
                self.generator_optimizer.zero_grad()
                data_fake = self.generator(z)

                generation_loss = -torch.mean(self.discriminator(data_fake))

                # regularization
                generator_regularization = torch.zeros_like(generation_loss)
                if self.generator.using_splines and self.config['generator_activation_fn_params']['lmbda'] > 0:
                    generator_regularization = self.config['generator_activation_fn_params']['lmbda'] * self.generator.TV2()

                generator_loss = generation_loss + generator_regularization
                generator_loss.backward()
                self.generator_optimizer.step()

                log['train_generator_loss'] = generator_loss.detach().cpu().item()

                if self.generator.using_splines:
                    spline_scaling_coeffs = torch.nn.utils.parameters_to_vector(spline_utils.get_spline_scaling_coeffs(self.generator))
                    log['generator_spline_scaling_coeff_mean'] = torch.mean(spline_scaling_coeffs).cpu().item()
                    log['generator_spline_scaling_coeff_std'] = torch.std(spline_scaling_coeffs).cpu().item()

                tbar.set_description('T ({}) | GeneratorLoss {:.5f} |'.format(epoch, log['train_generator_loss']))

            self.wrt_step = self.total_training_step * self.batch_size
            self.write_scalars_tb(log)

            self.total_training_step += 1

        return log

    def valid_epoch(self, epoch):
        
        self.discriminator.eval()
        
        with torch.no_grad():
            data_real = self.val_dataset.to(self.device) / 255
            data_fake = self.generator(torch.rand(10000, 62, device=self.device)).reshape(10000, 1, 28, 28)

            mean_wasserstein_distance = (torch.mean(self.discriminator(data_real)) - torch.mean(self.discriminator(data_fake)))

            # METRICS TO TENSORBOARD
            self.wrt_mode = 'val'
            self.writer.add_scalar(f'{self.wrt_mode}/Mean Wasserstein Distance', mean_wasserstein_distance, epoch)

            # Log parameters
            if epoch == self.epochs and self.generator.using_splines:
                j = 1
                for module in self.generator.modules_linearspline:
                    x = module.grid_tensor
                    y = module.lipschitz_coefficients
                    figures_list = []
                    for kk in range(x.shape[0]):
                        if kk % 16 == 0:
                            fig, ax = plt.subplots()
                            ax.grid()
                        ax.plot(x[kk,:].cpu().numpy(), y[kk,:].cpu().numpy())
                        if kk % 16 == 0:
                            figures_list.append(fig)
                            plt.close()
                    self.writer.add_figure('Generator Activation functions layer ' + str(j), figures_list, global_step=epoch)
                    j += 1

            if epoch == self.epochs and self.discriminator.using_splines:
                j = 1
                for module in self.discriminator.modules_linearspline:
                    x = module.grid_tensor
                    y = module.lipschitz_coefficients
                    figures_list = []
                    for kk in range(x.shape[0]):
                        if kk % 16 == 0:
                            fig, ax = plt.subplots()
                            ax.grid()
                        ax.plot(x[kk,:].cpu().numpy(), y[kk,:].cpu().numpy())
                        if kk % 16 == 0:
                            figures_list.append(fig)
                            plt.close()
                    self.writer.add_figure('Discriminator Activation functions layer ' + str(j), figures_list, global_step=epoch)
                    j += 1

            results = torch.zeros(224, 224)
            for i in range(8):
                for j in range(8):
                    results[i*28:(i+1)*28, j*28:(j+1)*28] = data_fake[i+8*j,0,...]
            self.writer.add_image('generation_results', results, epoch, dataformats='HW')


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)

    def save_checkpoint(self, epoch):
        state = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'config': self.config
        }

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint.pth'
        torch.save(state, filename)