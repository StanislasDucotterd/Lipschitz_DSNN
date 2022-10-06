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

Below we detail the model hyperparameters that can be controlled in the config file `configs/config_1d.json`.

```javascript
{
    "activation_fn_params": {
        "activation_type": "linearspline",
        "groupsort_groupsize": 5,
        "prelu_init": -1,
        "lipschitz_constrained": true,
        "spline_init": "relu",
        "spline_range": 0.5,
        "spline_scaling_coeff": true,
        "spline_size": 101,
        "lmbda": 1e-7
    },
    "dataset": {
        "function_type": "f1",
        "number_knots": 9,
        "testing_dataset_size": 10000,
        "training_dataset_size": 1000
    },
    "exp_name": "test",
    "log_dir": "1d_exps/ortho",
    "net_params": {
        "bias": true,
        "layer_sizes": [
            1,
            10,
            10,
            10,
            1
        ],
        "projection": "orthonormalize",
        "weight_initialization": "He_uniform"
    },
    "optimizer": {
        "lr_spline_coeffs": 5e-05,
        "lr_spline_scaling_coeffs": 0.0005,
        "lr_weights": 0.002
    },
    "seed": 5,
    "training_options": {
        "batch_size": 10,
        "epochs": 1000,
        "nbr_models": 25,
        "num_workers": 1
    }
}
```

Below we detail the model hyperparameters that can be controlled in the config file `configs/config_wasserstein.json`.

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