import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from dataloader.BSD500 import BSD500
from architectures.simple_cnn import SimpleCNN
from utils import metrics, utilities, spline_utils
import matplotlib.pyplot as plt
from layers.lipschitzconv2d import LipschitzConv2d
class TrainerDenoiser:
    """
    """
    def __init__(self, config, seed, device):
        self.config = config
        self.seed = seed
        self.device = device
        self.sigma = config['sigma']

        # Prepare dataset classes
        train_dataset = BSD500(config['train_dataloader']['train_data_file'])
        val_dataset = BSD500(config['val_dataloader']['val_data_file'])

        print('Preparing the dataloaders')
        # Prepare dataloaders 
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["train_dataloader"]["batch_size"], shuffle=config["train_dataloader"]["shuffle"], num_workers=config["train_dataloader"]["num_workers"], drop_last=True)
        self.batch_size = config["train_dataloader"]["batch_size"]
        self.val_dataloader = DataLoader(val_dataset, batch_size=config["val_dataloader"]["batch_size"], shuffle=config["val_dataloader"]["shuffle"], num_workers=config["val_dataloader"]["num_workers"])

        print('Building the model')
        # Build the model

        self.model = SimpleCNN(config['net_params'], **config['activation_fn_params'])
        self.model = self.model.to(device)
        for module in self.model.layers:
                if isinstance(module, LipschitzConv2d):
                    module.additional_parameters['largest_eigenvector'] = module.additional_parameters['largest_eigenvector'].to(device)
        
        print("Number of parameters in the model: ", self.model.get_num_params())

        print(self.model)
        
        # Set up the optimizer
        self.set_optimization()
        self.epochs = config["training_options"]['epochs']
        
        self.criterion = torch.nn.MSELoss(reduction='sum')

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

        self.total_number_channels = (config["net_params"]["num_layers"] - 1) * config["net_params"]["num_channels"]
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
        
        best_psnr = 0
        self.model.train()
        for epoch in range(self.epochs+1):

            if epoch >= self.epochs * 0.9:
                self.model.set_end_of_training()

            epoch_results = self.train_epoch(epoch)
            val_epoch_results = self.valid_epoch(epoch)
            self.scheduler.step()        
                
            # SAVE CHECKPOINT
            if val_epoch_results['val_psnr'] > best_psnr & epoch >= self.epochs * 0.9:
                best_psnr = val_epoch_results['val_psnr']
                self.save_checkpoint(epoch)
        
        self.writer.flush()
        self.writer.close()
        

    def train_epoch(self, epoch):
        """
        """
        self.model.train()

        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        log = {}
        for batch_idx, data in enumerate(tbar):
            data = data.to(self.device)
            noisy_data = data + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)

            self.optimizer.zero_grad()
                
            output = (noisy_data + self.model(noisy_data))/2.0

            # data fidelity
            data_fidelity = (self.criterion(output, data))/(self.batch_size)
                
            # regularization
            regularization = torch.zeros_like(data_fidelity)
            if self.model.using_splines and self.config['training_options']['lmbda'] > 0:
                regularization = self.config['training_options']['lmbda'] * self.model.TV2()

            total_loss = data_fidelity + regularization
            total_loss.backward()
            self.optimizer.step()
                
            log['train_loss'] = total_loss.detach().cpu().item()

            if self.total_training_step % (10 * 128 // self.batch_size)  == 0:
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
        loss_val = 0.0
        psnr_val = 0.0
        ssim_val = 0.0

        tbar_val = tqdm(self.val_dataloader, ncols=130, position=0, leave=True)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar_val):
                data = data.to(self.device)
                noisy_data = data + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)

                output = (noisy_data + self.model(noisy_data))/2.0

                loss = self.criterion(output, data)

                loss_val = loss_val + loss.cpu().item()
                out_val = torch.clamp(output, 0., 1.)
                psnr_val = psnr_val + utilities.batch_PSNR(out_val, data, 1.)
                ssim_val = ssim_val + utilities.batch_SSIM(out_val, data, 1.)
            
            # PRINT INFO
            loss_val = loss_val/len(self.val_dataloader)
            tbar_val.set_description('EVAL ({}) | MSELoss: {:.5f} |'.format(epoch, loss_val))

            # METRICS TO TENSORBOARD
            self.wrt_mode = 'val'
            self.writer.add_scalar(f'{self.wrt_mode}/loss', loss_val, epoch)
            psnr_val = psnr_val/len(self.val_dataloader)
            ssim_val = ssim_val/len(self.val_dataloader)
            self.writer.add_scalar(f'{self.wrt_mode}/Test PSNR', psnr_val, epoch)
            self.writer.add_scalar(f'{self.wrt_mode}/Test SSIM', ssim_val, epoch)

            log = {'val_loss': loss_val}
            log["val_psnr"] = psnr_val
            log["val_ssim"] = ssim_val

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
        
        return log


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