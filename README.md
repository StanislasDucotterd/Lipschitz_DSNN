# Lipschitz Constrained Deep Spline Neural Networks for Image Reconstruction
Master project carried out at the Biomedical imaging group (BIG) at EPFL during the fall 2021 semester under the supervision of Bohra Pakshal, Goujon Alexis & Perdios Dimitris.

#### Description
Image reconstruction is an ill-posed inverse problem, Plug-and-Play (PnP) priors is a framework that allows to solve this problem using convolutional neural network (CNN) denoisers. To guarantee the convergence of this algorithm, we need to impose some conditions on the denoisers. These conditions require us to train 1-Lipschitz CNNs. Constraining the Lipschitz constant of conventional ReLU CNNs leads to a severe drop in their performance. To mitigate this effect, we use a new model which benefits from learnable continuous piecewise linear spline activation functions; Deep Spline Neural Networks (DSNNs). When they are both Lipschitz-constrained, this architecture outperforms ReLU CNNs. DSNNs have not yet been exploited to the fullest of their potential. We propose new methods to improve the performance of DSNNs.

#### Requirements
The required packages:
- `pytorch`
- `torchvision`
- `opencv`
- `h5py`
- `cvxpy`
- `cvxpylayers`
- `qpsolvers` 
- `tqdm` 

You can install the exact environment I used with cudatoolkit 10.1 for the GPUs:

```bash
conda env create -f environment.yml
```

#### Training

You can train a model with the following command:

```bash
python train.py --model conv or fc --device cpu or gpu
```

You will need to download the BSD500 dataset to train convolutional denoisers. Information about the fully connected model is stored in `configs/config_fc.json`, information about the convolutional model is stored in `configs/config_conv.json`. Set `data_dir` to the dataset path in the config file in `configs/config_conv.json` when training the convolutiona model. You can train the fully connected model on synthetic data, there is four predefined 1-Lipschitz one-dimensional functions. You can also train it on random CPWL functions with an absolute derivative of one almost everywhere, you can choose the number of knots in the config file.

The log files and the `.pth` checkpoints will be saved in `saved\EXP_NAME\EXP_NAME_seed`, to monitor the training using tensorboard, you can run:

```bash
tensorboard --logdir directory of the tf.events files
```
#### Config file details️

Below we detail the model  parameters that can be controlled in the config file `configs/config_conv.json`.

```javascript
{
    "activation_fn_params": {
        "activation_type": "deepBspline",           // Can be multiple type of Deep Spline, GroupSort or ReLU
        "differentiable_projection": true,          // Make the activation 1-Lipschitz in a differentiable way
        "knot_threshold": 0,                        // Not used, keep it at 0
        "lipschitz_method": "slope_clipping",       // slope_norm, slope_clipping, pgd or l2_program
        "num_classes": 1,                                   
        "number_of_groups": -1,                     // choose the number of groups when using groupsort
        "save_memory": false,
        "spline_init": "leaky_relu",                // relu, leaky_relu, identity or even_odd
        "spline_range": 0.25,                        // Range where the activation functions are CPWL
        "spline_size": 51,                          // Number of spline coefficients
        "step_nbr": -1                              // only relevant when using pgd as lipschitz_method
    },
    "exp_name": "DSNN_sigma_5_layer_5_lambda_5",
    "logging_info": {
        "epochs_per_image": 100,                    // Plots the activation functions every X epochs
        "log_dir": "final_exps/",                   // Where the results of the experiment is stored
        "log_imgs": false,                          // Shows denoising results while training
        "log_iter": 1,                              // Get scalar information every X epochs
        "log_params": true,                         // Show distribution of parameters of neural networks
        "save_epoch": 100                           // Save all information at every X epochs
    },
    "net_params": {
        "alphas": true,                             // Adds additional parameters (alphas) to adjust the range
        "alphas_init": null,                        // Initialisation of the alphas, all equal to 1 by default
        "bias": true,   
        "kernel_size": 3,
        "num_channels": 32,
        "num_layers": 5,
        "padding": 1,
        "spectral_norm": true,                      // Makes the linear layer 1-Lipschitz
        "weight_initialization": "Xavier_uniform"         // He_uniform, He_normal, Xavier_uniform, Xavier_normal, custom_normal or identity
    },
    "optimizer": {
        "lr": 1e-05,                                // The lr when same optimizer for every parameter
        "same_optimizer": false,                    // puts weights, spline coeffs and alphas on same optim
        "lr_activation": 1e-06,                     // lr for spline coefficients
        "lr_alphas": 1e-05,                         // lr for alphas that adjust the range
        "lr_conv": 4e-05,                           // lr for weights and biases
        "type": "Adam"                              // optimizer type, Adam or SGD
    },
    "seed": 42,
    "sigma": 5,                                     // The additional gaussian noise has variance sigma/255
    "train_dataloader": {
        "batch_size": 128,
        "num_workers": 4,
        "shuffle": true,
        "train_data_file": "/path/to/training/set.h5"  // Location of training set
    },
    "training_options": {
        "centering": false,                          // Put the image in range [-0.5, 0.5] instead of [0, 1]
        "epochs": 200,
        "lipschitz_1_proj": false,                   // Non differentiable projection, only done if differentiable_projection is false, makes the activation 1-Lipschitz
        "lmbda": 1e-5,                               // Regularisation of the number of knots
        "mode": "half-averaged"                      // residual or half-averaged
    },
    "val_dataloader": {
        "batch_size": 1,
        "num_workers": 0,
        "shuffle": false,
        "val_data_file": "/path/to/validation/set.h5"   // Location of validation set
    }
}
```

Below we detail the additional model parameters that can be controlled in the config file `configs/config_fc.json`.

```javascript
{
    ...
    "net_params": {
        "bias": true,
        "layer_sizes": [1, 10, 10, 10, 1],          // Chooses the number of neurons of every layers
        "spectral_norm": true,
        "alphas": true,
        "weight_initialization": "Xavier_normal",
        "batch_norm": false                                                 
    ...
    "dataset" : {
        "training_dataset_size": 1000,              // Choose number of training point
        "testing_dataset_size": 10000,              // Choose number of validation point
        "function_type": "f2",                      // Choose between [f1, f2, f3, f4, random_spline]
        "nbr_models": 10,                           // Number of models trained, median or mean results will be reported
        "number_knots": 7                           // Number of knots of the random spline      
    },
    ...
}
```