# Single-Pipeline Point-Based Clothing Modeling

This repository integrates the inference steps for [*Generative Learning on Point-Based Modeling for Human Clothing*](https://github.com/UmairAhmadBaltoro/point_based_clothing) into one command. Given a `.jpg` image, the pipeline estimates a body model, generates a clothing segmentation mask, and renders the selected learned outfit on the person.

This is a reproduction and pipeline integration of the original project. The external repositories, model files, and pretrained weights still need to be installed or downloaded separately. The entry point is `piptest.py`; this repository does not train the underlying models from scratch.

## Pipeline

| Stage | Component | Result |
| --- | --- | --- |
| Body estimation | [ExPose](https://github.com/wyedoubleyou/expose_for_pbm) | SMPL-X mesh for the input image |
| Body conversion | [SMPL-X transfer](https://github.com/wyedoubleyou/yw_smplx) | SMPL parameters formatted for the clothing model |
| Human parsing | [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) | Clothing segmentation mask |
| Outfit inference | [point_based_clothing](https://github.com/UmairAhmadBaltoro/point_based_clothing) | Rendered image of the person wearing the selected outfit |

`piptest.py` coordinates these components. It copies the intermediate SMPL parameters and mask into `samples/internet_images/`, then calls the appearance inference in `hello2.py`. The provided appearance inference selects the `male-3-casual` outfit and writes a rendered PNG; edit `style_pid` and its corresponding checkpoint configuration in `hello2.py` to use another available outfit.

<img width="1280" height="611" alt="Point-based clothing pipeline overview" src="https://github.com/user-attachments/assets/713a0279-4c14-4f6e-a6b8-6e8ee3992927" />

## Setup 

### 1. Clone this repository and the external components

```bash
git clone https://github.com/wyedoubleyou/single_pipeline_point_based_modeling.git
cd single_pipeline_point_based_modeling

git clone https://github.com/wyedoubleyou/expose_for_pbm.git expose
git clone https://github.com/wyedoubleyou/yw_smplx.git smplx
git clone https://github.com/Gaoyiminggithub/Graphonomy.git Graphonomy
```

The expected top-level layout is:

```text
single_pipeline_point_based_modeling/
├── piptest.py
├── piptest2.py
├── hello2.py
├── expose/
├── smplx/
├── Graphonomy/
├── data/smpl_models/
├── samples/internet_images/
└── out/                  # downloaded weights and generated results
```

Install each external component using its own instructions. These directories are separate repositories and are not included in this repository's checkout. The integration expects the directory names shown above.

### 2. Prepare the Python environment

Guide for installation on CUDA, CUDA Toolkit, Conda Environment and libraries can be found at `Installation Guide.md`

The original development setup used Ubuntu 20.04, Python 3.9, an NVIDIA GPU, and PyTorch 1.13.0. Install a PyTorch build, CUDA runtime, PyTorch3D, and `nvdiffrast` that are mutually compatible with your GPU driver. Follow the installation instructions for [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for your environment.

### 3. Download the required models and weights

| Asset | Destination |
| --- | --- |
| [Pretrained point-based clothing assets](https://drive.google.com/drive/folders/1CnEZpaNvbiYvWrhK_i51Y67ODh8vKHeh?usp=sharing) | Place the downloaded `outfit_code` and `appearance` directories under `out/`. The current `hello2.py` expects `out/outfit_code/outfit_codes_psp.pkl` and an appearance checkpoint for `male-3-casual`. |
| [SMPL neutral model from SMPLify](https://smplify.is.tue.mpg.de/login.php) | After obtaining `SMPLIFY_CODE_V2.ZIP`, copy `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `data/smpl_models/SMPL_NEUTRAL.pkl`. Registration and the applicable model terms are handled by the provider. |
| [Additional model checkpoints](https://drive.google.com/file/d/1l9BKJyMo3tfSTh1u6NMFP9VBxXCMf-ZJ/view?usp=share_link) | Extract into `checkpoints/` as directed by the original project. |
| ExPose, SMPL-X transfer, and Graphonomy assets | Follow the instructions in their linked repositories. In particular, `piptest.py` expects `expose/data/conf.yaml`, `smplx/config_files/smplx2smpl.yaml`, and `Graphonomy/data/pretrained_model/universal_trained.pth`. |

Check that `expose/samples/test/`, `smplx/ply_folder/`, and `Graphonomy/img/` exist before running the script; `piptest.py` copies files into these directories. Its current ExPose stage also removes files already in `expose/samples/test/`.


## Run inference

1. Place one `.jpg` input image in `samples/internet_images/images/`. For example: `samples/internet_images/images/Pip1.jpg`.
2. From the repository root, activate your configured environment and run:

   ```bash
   python piptest.py Pip1
   ```

Use the image's basename without `.jpg` as the argument. The script expects a `.jpg` input and writes the Graphonomy mask as a `.png` file. It stores intermediate files in `samples/internet_images/segmentations/` and `samples/internet_images/smpl/`. The final rendered image is saved as `out/appearance/Pip1_output.png`.

The inference code uses CUDA (`cuda:0`) and has hard-coded paths and a selected outfit checkpoint. Run it from the repository root and adjust the paths or outfit settings in `hello2.py` if your files differ.


### Example outputs

Input images:

<img width="1280" height="438" alt="Example input images" src="https://github.com/user-attachments/assets/31c6b965-0dbf-4ef9-8db1-50cc4e4e8138" />

Segmentation masks from Graphonomy:

<img width="1280" height="415" alt="Example Graphonomy segmentation masks" src="https://github.com/user-attachments/assets/ce7c78e7-9cb5-42bd-afa6-e5f2ff2631a6" />

Body estimates from ExPose:

<img width="1195" height="404" alt="Example SMPL-X body estimates" src="https://github.com/user-attachments/assets/45c86af8-5dcc-443a-af8c-1973d0e94e1e" />

Output try-on: 

<img width="1280" height="333" alt="chp5_output" src="https://github.com/user-attachments/assets/404c1d80-2180-419e-b29d-d673760884d3" />


## Using a separate environment for SMPL-X transfer

If the SMPL-X transfer dependencies conflict with the main environment, edit `piptest.py` at the transfer stage: comment out the direct `python -m transfer_model ...` execution, and enable the provided `conda run -n py38 python .../piptest2.py` alternative. Replace `py38` with the name of the environment in which you installed the transfer model. `piptest2.py` runs the transfer command from inside the `smplx/` directory after `piptest.py` changes to it.

## Credits and citation

The clothing model comes from [point_based_clothing](https://github.com/UmairAhmadBaltoro/point_based_clothing). This repository integrates it with the linked ExPose, SMPL-X transfer, and Graphonomy implementations. Please consult those repositories and their papers for their licenses and citation instructions.

For the project describing this implementation:

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
