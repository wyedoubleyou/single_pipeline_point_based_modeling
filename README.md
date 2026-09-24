
This repository contains a reproduction and integration of the work presented in “Generative Learning on Point-Based Modeling for Human Clothing”, based on the original implementation from [point_based_clothing](https://github.com/UmairAhmadBaltoro/point_based_clothing). 

The original workflow relies on several external components for inference. In this reproduction, these components are integrated into a single end-to-end inference pipeline that automatically generates the required human segmentation masks and SMPL-X parameters before performing the point-based clothing modeling.

The pipeline integrates:
- **[Graphonomy](https://github.com/izakharkin/Graphonomy#inference-point_based_clothing)**  for human parsing and segmentation mask generation.
- **[ExPose](https://github.com/vchoutas/expose)** to estimate SMPL-X body parameters from the input image.
- **[SMPL-X Transfer Model](https://github.com/vchoutas/smplx)** SMPL-X model processing and parameter transfer to SMPL model.
- **[point_based_clothing](https://github.com/UmairAhmadBaltoro/point_based_clothing)** for final point-based human clothing generation.


## Pipeline Overview

The inference pipeline can be summarized as:

```text
Input Image
    │
    ├── Graphonomy
    │      └── Human Segmentation Mask
    │
    ├── ExPose
    │      └── SMPL-X Parameters
    │
    └── SMPL-X Transfer
           │
           ▼
Point-Based Clothing Model
           │
           ▼
Generated Clothed Human
```

Compared with the original implementation, this repository combines the previously separate preprocessing and inference stages into a **unified pipeline**, reducing the need to manually generate segmentation masks and SMPL-X parameters.

<img width="1280" height="611" alt="5 1" src="https://github.com/user-attachments/assets/713a0279-4c14-4f6e-a6b8-6e8ee3992927" />

# Installation Guide 

This project is preferable to conduct in Ubuntu Desktop. The Ubuntu Desktop version and other software layes that I use is listed below. The installation guide for the CUDA, CuDNN, conda environment, and python libraries are given as below.  

## Ubuntu & Driver

- Ubuntu Version: Ubuntu 20.04.6 LTS (Focal Fossa)
- NVIDIA Driver: 470.223.02
  
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

Here are some libraries required to installed manually in order to run the project code. Pytorch3d version 1.13.0 is required. 

### Create conda environment with python version 3.9 and installed Pytorch3d 
- [Source page](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- `conda create -n pytorch3d python=3.9`
- `conda activate pytorch3d`
- `conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia`
- `conda install -c fvcore -c iopath -c conda-forge fvcore iopath`
- `conda install -c bottler nvidiacub` 
- Use **Nightly** version `conda install pytorch3d -c pytorch3d-nightly`

### Download Scatter (optional) *if dont have library error
- [Source page](https://data.pyg.org/whl/torch-1.10.0+cu102.html)
- Library to download at the webpage
  - torch_cluster-1.6.0+pt113cu116-cp39-cp39-linux_x86_64.whl
  - torch_sparse-0.6.15+pt113cu116-cp39-cp39-linux_x86_64.whl
  - torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
  - torch_spline_conv-1.2.1+pt113cu116-cp39-cp39-linux_x86_64.whl
  - pyg_lib-0.2.0+pt113cu116-cp39-cp39-linux_x86_64.whl

- In the **Download** directory 
  - `pip install torch_cluster-1.6.0+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_sparse-0.6.15+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_spline_conv-1.2.1+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install pyg_lib-0.2.0+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch-geometric`


### Other dependencies

`pip install opencv-python`

`pip install pyyaml`

`pip install imageio`

`pip install pandas`

`pip install munch`

`pip install smplx`

`pip install open3d`

`pip install imgaug`

`pip install git+https://github.com/DmitryUlyanov/yamlenv`

`pip install huepy`

`pip install kornia==0.6.0`

  ``` shell 
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast 
pip install .
  ```
`pip install Ninja`

`pip install chumpy`



# To run the code, 

## Clone original repo
- `git clone https://github.com/UmairAhmadBaltoro/point_based_clothing.git'
- `cd point_based_clothing`


## Download pre-trained data
- Download pre-trained model from [Google Drive ](https://drive.google.com/drive/folders/1CnEZpaNvbiYvWrhK_i51Y67ODh8vKHeh?usp=sharing) and save `outfit_code` and `appearance` to `out\` folder
- Download the SMPL neutral model from [SMPLify project page](https://smplify.is.tue.mpg.de/login.php): 
  - Register, go to the `Downloads` section, download `SMPLIFY_CODE_V2.ZIP`, and unpack it;
  - Move `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `data/smpl_models/SMPL_NEUTRAL.pkl`.
- Download models checkpoints (~570 Mb): [Google Drive](https://drive.google.com/file/d/1l9BKJyMo3tfSTh1u6NMFP9VBxXCMf-ZJ/view?usp=share_link) and place them to the `checkpoints/` folder;


## Sub-model,  

- Install [graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) in the `point_based_clothing/` directory. The directory name should be `Graphonomy`
- Installation of [ExPose](https://github.com/wyedoubleyou/expose_for_pbm) in `point_based_clothing/` directory, ExPose's directory name should be `expose`
- Installation of [SMPL-X](https://github.com/wyedoubleyou/yw_smplx.git) in `point_based_clothing/` directory, SMPL-X's directory name should be `smplx`


## Inference Process

- store your RGB image in `samples/internet_images/images/` folder, make sure the image is in **`.jpg`** format to prevent duplication, as the clothing segmentation mask from Graphonomy is in `.png` format
- the `<rgb_file_name>` is the name of the RGB image without extension. E.g. "Pip1.jpg" the `<rgb_file_name>` will be "Pip1". Example of the command are as below:
  
  ``` shell 
  python piptest.py <rgb_file_name>
  
  ```
  
- To run the code, you can try with command:

  ``` shell 
  python piptest.py Pip1
  
  ```

- the whole inference process will be take around 1-2 minutes, and the final appearance result will be store in `out/appearance/` folder

Input Images: 
<img width="1280" height="438" alt="chp5_mc_crop" src="https://github.com/user-attachments/assets/31c6b965-0dbf-4ef9-8db1-50cc4e4e8138" />

Segmentation mask generated using Graphonomy: 
<img width="1280" height="415" alt="chp5_grp_mc" src="https://github.com/user-attachments/assets/ce7c78e7-9cb5-42bd-afa6-e5f2ff2631a6" />

SMPL-X model generated using Expose: 
<img width="1195" height="404" alt="chp5_sm_mc" src="https://github.com/user-attachments/assets/45c86af8-5dcc-443a-af8c-1973d0e94e1e" />


## In case not able to run all model in one conda environment 

- In `piptest.py` comment out line 80 & 81, then uncomment and modify line 85 according to your needs
- For my case, i use `py38` (another conda environment) and `piptest2.py` for converting SMPL-X to SMPL parameter to prevent library conflict error. The command for the transfer model is store in `piptest2.py`

  ``` shell 
  subprocess.run(f'conda run -n py38 python {mydir+"/"}piptest2.py', shell=True)
  ```

## Citation
More details can be found in my FYP published paper:
```bibtex
@inproceedings{wei2024implementation,
  title={Implementation of Cloth Estimation in 2D-3D Human Body Regression Model},
  author={Wei, Fung Yi and Lim, King Hann and Phang, Jonathan Then Sien and Pang, Po Ken},
  booktitle={2024 International Conference on Green Energy, Computing and Sustainable Technology (GECOST)},
  pages={224--228},
  year={2024},
  organization={IEEE}
}
```
