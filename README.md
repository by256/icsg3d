# Inorganic Crystal Structure Generation in 3D (ICSG3D)
A Generative deep learning pipeline for 3D crystal structures and property prediction. Soure code associated with the paper ...

![Example crystals generated with our system](images/fig-crystals.pdf)

# Installation
1. Clone the git repository
> git clone https://github.com/by256/icsg3d
2. Install requirements
> python3 -m pip install -r requirements.txt

# Getting Data
The first stage of the pipeline is to retrieve crystallographic information files (CIFs) to train the deep learning pipeline. In theory these can be from any source, but by default we use the materialsproject API.

For example, to retrieve all CIFs for ternary cubics (ABC):
> python3 query_matproj.py --anonymous_formula="{'A': 1.0, 'B': 1.0, 'C':1.0}" --system=cubic --name=ternary_cubic

This will create a data/ternary_cubic folder containing the cifs and a csv with associated properties

# Creating the natwork inputs
The various network input matrices can be created by
> mpiexec -n 4 python3 create_matrices.py --name=ternary_cubic

# Train the UNET
Trai the unet for as many epochs as needed
> python3 train_unet.py --name ternary_cubic --samples 10000 --epochs 50

# Train the VAE
Make sure you train the VAE second (as it uses the unet as a DFC perceptual model)
> python3 train_vae.py --name ternary_cubic --nsamples 1000 --epochs 250

# View some results
1. Interpolations in vae latent space
> python3 interpolate.py --name ternary_cubic

2. Whole pipeline plots
> python3 view_results.py --name ternary_cubic

3. Evaluate coordinates and lattice params
> python3 eval.py --name ternary_cubic

# Generate new samples
> python3 generate.py --name ternary_cubic --nsamples 1000 --base mp-1234

This will create a new directory in Results where you will find Cifs, density matrices, species matrices and properties.