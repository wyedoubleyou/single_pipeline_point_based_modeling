# Point-Based Modeling of Human Clothing

Modify version of official PyTorch code repository of the paper "Point-Based Modeling of Human Clothing" (accepted to ICCV, 2021).

<p align="center">
  <b>
  <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Zakharkin_Point-Based_Modeling_of_Human_Clothing_ICCV_2021_paper.html">Paper</a> 
  | <a href="https://www.ilia.ai/research/point-based-clothing">Project page</a>
  | <a href="https://youtu.be/kFrAu415kDU">Video</a>
    </b>
</p>


# Installation Guide 

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

Fail to run docker container from the original project. Here are some libraries required to installed manually in order to run the project code. 

### Download Pytorch3d 
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

- Run the **Download** directory 
  - `pip install torch_cluster-1.6.0+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_sparse-0.6.15+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl`
  - `pip install torch_spline_conv-1.2.1+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install pyg_lib-0.2.0+pt113cu116-cp39-cp39-linux_x86_64.whl`
  - `pip install torch-geometric`


### Additional library 
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


# Setup 

## Clone repo
- [Source page](https://github.com/SamsungLabs/point_based_clothing)
- Prerequisites: your nvidia driver should support cuda 10.2, Windows or Mac are not supported.
- Clone repo:
  - `git clone https://github.com/izakharkin/point_based_clothing.git`
  - `cd point_based_clothing`


## Download data for Point-based modeling Inference 

- Download pre-trained model from [Google Drive ](https://drive.google.com/drive/folders/1CnEZpaNvbiYvWrhK_i51Y67ODh8vKHeh?usp=sharing) and save `outfit_code` and `appearance` to `out\` folder
- Download the SMPL neutral model from [SMPLify project page](https://smplify.is.tue.mpg.de/login.php): 
  - Register, go to the `Downloads` section, download `SMPLIFY_CODE_V2.ZIP`, and unpack it;
  - Move `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `data/smpl_models/SMPL_NEUTRAL.pkl`.
- Download models checkpoints (~570 Mb): [Google Drive](https://drive.google.com/file/d/1l9BKJyMo3tfSTh1u6NMFP9VBxXCMf-ZJ/view?usp=share_link) and place them to the `checkpoints/` folder;
- **(Optional: this is only for training)** Download a sample data we provide to check the appearance fitting (~480 Mb): [Google Drive](https://drive.google.com/file/d/1QBZu9SLNoXdhLdTYABU-_KinjUAoQfw-/view?usp=share_link), unpack it, and place `psp/` folder to the `samples/` folder.


## Graphonomy 

- [Source page](https://github.com/izakharkin/Graphonomy#inference-point_based_clothing)
- In the `point_based_clothing/` directory, `git clone https://github.com/Gaoyiminggithub/Graphonomy`
- Graphonomy's directory name should be `Graphonomy`
- Library installation according to [original fork](https://github.com/Gaoyiminggithub/Graphonomy) of Graphonomy to obtain clothing segmentation mask in our format;
- create `output/` folder
- Get the [universal model weights](https://drive.google.com/file/d/1jO35B5GVQfJQWuL_KjYkVdc9bmRXnQJ4/view) and place them in `data/pretrained_model/` folder. We modified the original inference.py script for it to output the segmentation mask in the format of the point_based_clothing repo:

```shell
# Example of inference (test this seperately first to see if Graphonomy is working)
python exp/inference/inference.py  \
--loadmodel data/pretrained_model/universal_trained.pth \
--img_path ./img/messi.jpg \
--output_path ./output/ \
--output_name /output_file_name
```


## ExPose 

- [Source page](https://github.com/vchoutas/expose)
- Install `expose` in the `point_based_clothing/` directory, ExPose's directory name should be `expose`
- Installation of files and library according to [my ExPose fork](https://github.com/wyedoubleyou/expose_for_pbm) to obtain the SMPL-parameters (3D body pose and shape ground truth).


## SMPL-X Transfer Model

- [Source page](https://github.com/vchoutas/smplx)
- Install `smplx` in the `point_based_clothing/` directory, SMPL-X's directory name should be `smplx`
- Installation of files and library according to [my smplx fork](https://github.com/wyedoubleyou/yw_smplx.git) to transfer the SMPL-X into SMPL parameter. 


## Point-based Modeling Inference process 

- store your human model RGB image in `samples/internet_images/images/` folder, make sure the image is in **`.jpg`** format to prevent duplication, as the clothing segmentation mask from Graphonomy is in `.png` format
- the `<rgb_file_name>` is the name of the RGB image without extension. E.g. "Pip1.jpg" the `<rgb_file_name>` will be "Pip1". Example of the single pipeline command are as below:

  ``` shell 
  python piptest.py <rgb_file_name>
  
  ```
  
- `Pip1.jpg` is provided, you can try the code with command:
- 
  ``` shell 
  python piptest.py Pip1
  
  ```
- the whole inference process will be take around 1-2 minutes, and the final appearance result will be store in `out/appearance/` folder


## In case not able to run all model in one conda environment 

- In `piptest.py` comment out line 80 & 81, then uncomment and modify line 85 according to your needs
- For my case, i use `py38` (another conda environment) and `piptest2.py` for converting SMPL-X to SMPL parameter to prevent library conflict error. The command for the transfer model is store in `piptest2.py`

  ``` shell 
  subprocess.run(f'conda run -n py38 python {mydir+"/"}piptest2.py', shell=True)
  ``` 
