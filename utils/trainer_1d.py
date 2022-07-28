import torch
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from architectures.simple_fc import SimpleFC
from utils import metrics, utilities, spline_utils
import matplotlib.pyplot as plt
from layers.lipschitzlinear import LipschitzLinear
from dataloader.Function_1D import (Function1D, generate_testing_set,
     slope_1_ae, slope_1_flat, cosines, sawtooth)


class Trainer1D:
    """
    """
    def __init__(self, config, seed, device):
        self.config = config
        self.seed = seed
        self.device = device

        # Prepare dataset
        if config["dataset"]["function_type"] == "slope_1_ae":
            function = lambda x: slope_1_ae(x, config["dataset"]["number_knots"], self.seed)
        elif config["dataset"]["function_type"] == "slope_1_flat":
            function = lambda x: slope_1_flat(x, config["dataset"]["number_knots"], self.seed)
        elif config['dataset']['function_type'] == 'cosines':
            function = lambda x: cosines(x)
        elif config['dataset']['function_type'] == 'sawtooth':
            function = lambda x: sawtooth(x)
        else:
            raise NameError('Invalid Function Type')

        train_dataset = Function1D(function, config["dataset"]["training_dataset_size"], config["training_options"]["nbr_models"], seed)
        self.val_dataset = generate_testing_set(function, config["dataset"]["testing_dataset_size"])

        print('Preparing the dataloaders')
        # Prepare dataloaders 
        self.batch_size = config["training_options"]["batch_size"]
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=config["training_options"]["num_workers"], shuffle=True, drop_last=True)

        print('Building the model(s)')
        # Build the model

        self.models = []
        self.nbr_models = config["training_options"]["nbr_models"]
        for i in range(self.nbr_models):
            self.models.append(SimpleFC(config['net_params'], **config['activation_fn_params']).to(self.device))

        print("Number of models : ", self.nbr_models)
        print("Number of parameters in the model(s): ", self.models[0].get_num_params())
        print(self.models[0])
        
        # Set up the optimizer
        self.set_optimization()
        self.epochs = config["training_options"]['epochs']
        self.testing_dataset_size = config["dataset"]["testing_dataset_size"]

        #Stats to save about the models
        self.test_mse = torch.zeros((self.nbr_models, 1))
        
        self.criterion = torch.nn.MSELoss(reduction='sum')

        # CHECKPOINTS & TENSOBOARD
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
        """ """
        self.optimizers = []

        for i in range(self.nbr_models):
            params_list = [{'params': spline_utils.get_no_spline_coefficients(self.models[i]), \
                            'lr': self.config["optimizer"]["lr_weights"]}]
            if self.models[0].using_splines:
                params_list.append({'params': spline_utils.get_spline_coefficients(self.models[i]), \
                                    'lr': self.config["optimizer"]["lr_spline_coeffs"]})

                if self.config["activation_fn_params"]["spline_scaling_coeff"]:
                    params_list.append({'params': spline_utils.get_spline_scaling_coeffs(self.models[i]), \
                                        'lr': self.config["optimizer"]["lr_spline_scaling_coeffs"]})
            self.optimizers.append(torch.optim.Adam(params_list))


    def train(self):

        min_val_loss = 1e8
        for epoch in range(self.epochs+1):
            epoch_results = self.train_epoch(epoch)
            val_epoch_results = self.valid_epoch(epoch)
            if val_epoch_results < min_val_loss:
                self.save_checkpoint(epoch)
                min_val_loss = val_epoch_results
        
        self.writer.flush()
        self.writer.close()
        

    def train_epoch(self, epoch):
        """
        """
        for i in range(self.nbr_models):
            self.models[i].train()

        tbar = tqdm(self.train_dataloader, ncols=135)
        log = {}
        for batch_idx, data in enumerate(tbar):

            input_data = data[0].to(self.device)
            target_data = data[1].to(self.device)

            for i in range(len(self.optimizers)):
                self.optimizers[i].zero_grad()
            
            outputs = []
            for i in range(self.nbr_models):
                outputs.append(self.models[i](input_data[:, i, :]))

            # data fidelity
            data_fidelities = []
            for i in range(self.nbr_models):
                data_fidelities.append(self.criterion(outputs[i], target_data[:, i, :]) / self.batch_size)
            
            # regularization
            regularizations = self.nbr_models * [torch.zeros_like(data_fidelities[0])]
            if self.models[0].using_splines and self.config['training_options']['lmbda'] > 0:
                for i in range(self.nbr_models):
                    regularization = self.config['training_options']['lmbda'] * self.models[i].TV2()
                    regularizations[i] = regularization

            total_losses = []
            for i in range(self.nbr_models):
                total_losses.append(data_fidelities[i] + regularizations[i])
            for i in range(self.nbr_models):
                total_losses[i].backward()

            self.optimizer_step()

            mean_total_loss = 0
            for i in range(self.nbr_models):
                mean_total_loss += total_losses[i].detach().cpu().item()  
            log['Mean train loss'] = mean_total_loss / self.nbr_models

            if self.total_training_step % (100 // self.batch_size)  == 0:

                if self.config["activation_fn_params"]["spline_scaling_coeff"] & self.models[0].using_splines:
                    spline_scaling_coeff_mean = 0
                    spline_scaling_coeff_mean_std = 0
                    for i in range(len(self.models)):
                        spline_scaling_coeffs = torch.nn.utils.parameters_to_vector(spline_utils.\
                                                get_spline_scaling_coeffs(self.models[i]))
                        spline_scaling_coeff_mean += torch.mean(spline_scaling_coeffs).cpu().item() / self.nbr_models
                        spline_scaling_coeff_mean_std += torch.std(spline_scaling_coeffs).cpu().item() / self.nbr_models
                    log['spline_scaling_coeff_mean'] = spline_scaling_coeff_mean
                    log['spline_scaling_coeff_mean_std'] = spline_scaling_coeff_mean_std
                self.wrt_step = self.total_training_step * self.batch_size
                self.write_scalars_tb(log)

            tbar.set_description('T ({}) | TotalLoss {:.8f} |'.format(epoch, log['Mean train loss']))
            self.total_training_step += 1

        return log

    
    def optimizer_step(self):
        """ """
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()


    def valid_epoch(self, epoch):
        
        for i in range(self.nbr_models):
            self.models[i].eval()

        with torch.no_grad():
            losses = np.zeros((self.nbr_models, 1))
            preds = np.zeros((self.nbr_models, self.testing_dataset_size))
            for i in range(self.nbr_models):
                pred = self.models[i](self.val_dataset[0].to(self.device))
                loss = self.criterion(pred, self.val_dataset[1].to(self.device)).cpu().item()
                losses[i] = loss
                preds[i,:] = pred[:,0].cpu().numpy()

            median_loss = np.median(losses) / self.testing_dataset_size
            min_loss = np.min(losses) / self.testing_dataset_size
            self.writer.add_scalar('Validation/Median Loss', median_loss, epoch)             
            self.writer.add_scalar('Validation/Min Loss', min_loss, epoch)
            max_loss_index = int(np.argmax(losses))
            min_loss_index = int(np.argmin(losses))
            median_loss_index = int(np.argsort(losses)[len(losses)//2])
            indices = [max_loss_index, median_loss_index, min_loss_index]
            titles = ['Maximum Loss', 'Median Loss', 'Minimum Loss']

            if epoch == self.epochs:
                self.test_mse = torch.tensor(losses)

            if epoch %  100== 0:
                figures_list = []
                for i in range(3):
                    fig, ax = plt.subplots()
                    ax.grid()
                    pred = preds[indices[i]]
                    input_ = self.val_dataset[0].cpu().numpy()
                    target = self.val_dataset[1].cpu().numpy()
                    ax.plot(input_, target)
                    ax.plot(input_, pred.T)
                    fig.suptitle(titles[i], fontsize=18)
                    figures_list.append(fig)
                    plt.close()
                self.writer.add_figure(f'Max, Median and Min Loss predictions', figures_list, global_step=epoch)

            if epoch == self.epochs and self.models[0].using_splines:
                j = 1
                for module in self.models[min_loss_index].modules_linearspline:
                    x = module.grid_tensor
                    y = module.lipschitz_coefficients
                    figures_list = []
                    for kk in range(x.shape[0]):
                        fig, ax = plt.subplots()
                        ax.grid()
                        ax.plot(x[kk,:].cpu().numpy(), y[kk,:].cpu().numpy())
                        figures_list.append(fig)
                        plt.close()
                    self.writer.add_figure('Best Model Activation functions layer ' + str(j), figures_list, global_step=epoch)
                    j += 1
        return min_loss


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'Training/{k}', v, self.wrt_step)

    def save_checkpoint(self, epoch):
        state_dicts = []
        for i in range(self.nbr_models):
            state_dicts.append(self.models[i].state_dict())
        state = {
            'epoch': epoch,
            'state_dicts': state_dicts,
            'test_mse': self.test_mse,
            'config': self.config
        }

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint_best_epoch.pth'
        torch.save(state, filename)