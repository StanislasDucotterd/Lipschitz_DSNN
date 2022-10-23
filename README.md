# Improving Lipschitz-Constrained Neural Networks by Learning Activation Functions
Implementation of experiments done in : https://arxiv.org/abs/number.number

#### Description
Lipschitz-constrained neural networks have several advantages compared to unconstrained ones and can be applied to various different problems. Consequently, they have recently attracted considerable attention in the deep learning community. However, it has been shown both theoretically and empirically that networks with ReLU activation functions perform poorly in this context. On the contrary, neural networks with learnable 1-Lipschitz linear splines are known to be more expressive in theory. We propose an efficient method to train such 1-Lipschitz deep spline neural networks. Our numerical experiments for a variety of tasks show that our trained networks match or outperform networks with activation functions specifically tailored towards Lipschitz-constrained architectures.

#### Requirements
The required packages:
- `pytorch`
- `torchvision`
- `opencv`
- `h5py`
- `cvxpy`
- `cvxpylayers`
- `tqdm` 

You can install the exact environment I used with cudatoolkit 10.1 for the GPUs:

```bash
conda env create -f environment.yml
```

#### Training

You can train a model with the following command:

```bash
python train.py --exp 1d/wasserstein/denoising --config path/to/config --device cpu/cuda:n
```

#### Config file details️

Information about the hyperparameters of the three experiments can be found in the config folder. 

Below we detail the model hyperparameters for the denoising experiment that can be controlled in the config file `configs/config_denoising.json`.

```javascript
{
    "activation_fn_params": {
        "activation_type": "linearspline",      // choose relu/absolute_value/prelu/groupsort/householder/linearspline
        "groupsort_groupsize": 2,              
        "prelu_init": -1,                       // number in [-1, 1] or "maxmin" (half with 1 and other half with -1)
        "lipschitz_constrained": true,
        "spline_init": "identity",              // can be identity/relu/absolute_value/maxmin
        "spline_range": 0.1,
        "spline_size": 51,                      // number of linear regions +1
        "lmbda": 1e-6                           // TV2 reg 
    },
    "exp_name": "name_of_exp",
    "log_dir": "denoising_exps/sigma_5",
    "net_params": {
        "bias": true,
        "kernel_size": 3,
        "num_channels": 64,
        "num_layers": 5,
        "padding_mode": "zeros",
        "projection": "spectral_norm",          // how to make conv layer 1-Lipschitz no_projection/spectral_norm
        "signal_size": 256,                     // size of the eigenimage to estimate spectral norm in power iter
        "weight_initialization": "identity"     // He_uniform/He_normal/Xavier_uniform/Xavier_normal/identity
    },
    "optimizer": {                              
        "lr_spline_coeffs": 1e-06,
        "lr_spline_scaling_coeffs": 1e-05,
        "lr_weights": 4e-05
    },
    "seed": 42,
    "sigma": 5,                                 // noise level
    "training_options": {
        "epochs": 20,
        "batch_size": 4,
        "num_workers": 1,
        "train_data_file": "path/to/train.file",
        "val_data_file": "path/to/val.file"
    }
}
```

Below we detail the model hyperparameters that can be controlled in the config file `configs/config_wasserstein.json`.

```javascript
{
    "net_params": {
        "bias": true,
        "bjorck_iter": 25,                      // nb of iters for the bjorck algorithm to orthonormalize weight matrices
        "layer_sizes": [
            784,
            1024,
            1024,
            1
        ],
        "projection": "orthonormalize",
        "weight_initialization": "orthonormal" // 
    },
    "optimizer": {
        "lr_spline_coeffs": 5e-05,
        "lr_spline_scaling_coeffs": 0.0005,
        "lr_weights": 0.002
    },
    "seed": 42,
    "training_options": {
        "batch_size": 4096,
        "epochs": 1600,
        "num_workers": 1,
        "train_dataset_file": "data/mnist/train.pt",
        "val_dataset_file": "data/mnist/val.pt"
    }
}
```

Below we detail the model hyperparameters that can be controlled in the config file `configs/config_denoiser.json`.

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