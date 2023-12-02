# point-based-clothing
Generative learning on point-based modeling for human clothing

# Installation Guide 

## Ubuntu & Driver
- Ubuntu Version: Ubuntu 20.04.6 LTS (Focal Fossa)

- NVIDIA Driver: 470
  
### CUDA 11.4 

  - [Link for CUDA Toolkit 11.4 Download](https://developer.nvidia.com/cuda-11-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local) 
  - `wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run`
  - `sudo sh cuda_11.4.0_470.42.01_linux.run`
  - `gedit ~/.bashrc`


- Conda Initialize
    - `export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}`
    - `export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`
    - `echo $PATH`

### CuDNN v8.2.2
- [Download cuDNN v8.2.2 (July 6th, 2021), for CUDA 11.4](https://developer.nvidia.com/rdp/cudnn-archive)
- `tar -xvf cudnn-11.4-linux-x64-v8.2.2.26.tgz`
- `sudo cp cuda/include/cudnn*.h /usr/local/cuda/include`
- `sudo cp -P cuda/lib/libcudnn* /usr/local/cuda/lib64`
- `sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*`


## Python Library 

Fail to run docker container from the original project. Here are some libraries required to installed manually in order to run the project code. 

### Download Pytorch3d 
- [Source page](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- `conda create -n pytorch3d python=3.9`
- `conda activate pytorch3d`
- `conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia`
- `conda install -c fvcore -c iopath -c conda-forge fvcore iopath`
- `conda install -c bottler nvidiacub`
- Use **Nightly** version `conda install pytorch3d -c pytorch3d-nightly`

### Download Scatter (optional *if dont have library error)
- [Source page](https://data.pyg.org/whl/torch-1.10.0+cu102.html)
- Library to download at the webpage
  - torch_cluster-1.6.0+pt113cu116-cp39-cp39-linux_x86_64.whl
  - torch_sparse-0.6.15+pt113cu116-cp39-cp39-linux_x86_64.whl
  - torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
  - torch_spline_conv-1.2.1+pt113cu116-cp39-cp39-linux_x86_64.whl
  - pyg_lib-0.2.0+pt113cu116-cp39-cp39-linux_x86_64.whl

- Run the **Download** directory 
  - `pip install torch_cluster-1.6.0+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_sparse-0.6.15+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_spline_conv-1.2.1+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install pyg_lib-0.2.0+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch-geometric`


### Additional library 

# Setup 

## Clone repo

- Prerequisites: your nvidia driver should support cuda 10.2, Windows or Mac are not supported.
- Clone repo:
  - `git clone https://github.com/izakharkin/point_based_clothing.git`
  - `cd point_based_clothing`
  - `git submodule init && git submodule update`

## Download data 

- Download pre-trained model from [Google Drive ][https://drive.google.com/drive/folders/1CnEZpaNvbiYvWrhK_i51Y67ODh8vKHeh?usp=sharing] and save `outfit_code` and `appearance` to `out\` folder
- Download the SMPL neutral model from [SMPLify project page](https://smplify.is.tue.mpg.de/login.php): 
  - Register, go to the `Downloads` section, download `SMPLIFY_CODE_V2.ZIP`, and unpack it;
  - Move `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `data/smpl_models/SMPL_NEUTRAL.pkl`.
- Download models checkpoints (~570 Mb): [Google Drive](https://drive.google.com/file/d/1l9BKJyMo3tfSTh1u6NMFP9VBxXCMf-ZJ/view?usp=share_link) and place them to the `checkpoints/` folder;
- Download a sample data we provide to check the appearance fitting (~480 Mb): [Google Drive](https://drive.google.com/file/d/1QBZu9SLNoXdhLdTYABU-_KinjUAoQfw-/view?usp=share_link), unpack it, and place `psp/` folder to the `samples/` folder.



