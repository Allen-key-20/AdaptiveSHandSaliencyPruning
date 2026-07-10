# Adaptive Spherical-Harmonic Allocation and Saliency Pruning for Memory-Efficient 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) has become a popular method for efficient and high-quality novel view synthesis. However, its training process demands excessive GPU memory, primarily due to uniform storage of high-order spherical-harmonic coefficients and rapid increases in the number of Gaussians. To address this, we introduce a memory-efficient approach that dynamically adjusts spherical-harmonic orders based on the expressive needs of each Gaussian and employs a saliency-based pruning strategy to remove redundant Gaussians. Our method reduces the peak GPU memory consumption during training by 29.6\%, 29.8\%, and 22.4\% on the Mip-NeRF360, Tanks\&Temples, and Deep Blending datasets, respectively, while maintaining rendering quality comparable to the original approach.

## Setup

### Local Setup

Our default, experiments were conducted on Ubuntu22.04 and cuda11.8


```shell
git clone https://github.com/Allen-key-20/[Efficient-Memory-Management-in-3D-Gaussian-Splatting.gi](https://github.com/Allen-key-20/AdaptiveSHandSaliencyPruning --recursive

cd Efficient-Memory-Management-in-3D-Gaussian-Splatting

conda env create --file environment.yml
```

### Dataset
You can find three datasets in [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) . Put the downloaded dataset into the "data" folder

```
data
  ├── Mip-NeRF360
  │   ├── bicycle
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
  ├── Deep Blending
  │   ├── playroom
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
  ├── Tanks&Temples
  │   ├── truck
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
```

### Running

Train a single scene. Set `scene = ' '` in the `run.py` file to the name of the scenario to be trained. And set `dateset = ' '` to which the scene belongs.
```shell
python run.py 
```

Train the entire dataset. Train the default three datasets.
```shell
python run_folder.py 
```
### Note

Our code is modified based on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) . The simpleknn in the submodules folder is also from this project.
An initial open-source version of our method is already available, which demonstrates the core methodology and enables reproduction of the main experimental results.    To ensure code quality and usability, we are adopting a progressive open-source strategy: more complete and optimized implementations will be released as the development stabilizes.
