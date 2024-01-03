# AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose (AAAI2024)
Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu1, Xu Wang, Li Chen, Chao Long, Feida Zhu, Daniel K. Du, Shilei Wen

| [Project Page](https://avatarverse3d.github.io/) | [Github](https://github.com/bytedance/AvatarVerse) | [Paper](https://arxiv.org/abs/2308.03610)|

![Avatar](./figs/avatarverse.gif)


## Installation
Prepare running enviroment:
- You must have an NVIDIA graphics card with at least 20GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.7`.
- (Optional, Recommended) Create a virtual environment:

```sh
python3 -m virtualenv venv
. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```

- Install `PyTorch >= 1.12`. We have tested on `torch-1.10.0+cu113` and `torch2.0.1+cu117`, but other versions should also work fine.

- Install [`kaolin`](https://github.com/NVIDIAGameWorks/kaolin) library:
```
# Replace TORCH_VERSION and CUDA_VERSION with your torch / cuda versions
# e.g. https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu117.html

pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VERSION}_cu{CUDA_VERSION}.html
```

- Install [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter) library. Change the version according to your torch version, more information can be found [here](https://github.com/rusty1s/pytorch_scatter):
```
pip3 install -v torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
- run the `setup_env.sh` scripts.
```sh
bash setup_env.sh
```

- optional: install `xformers` library, the installation of xformers may take some time(sometimes more than 30 minutes):
```
pip3 install -v git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```


## Model Card
Our models are provided on the [Huggingface Model Page](https://huggingface.co/liqingzju/AvatarVerse/) with the OpenRAIL license.

| Model      | Base Model | Resolution |
| ----------- | ----------- | ----------- |
| controlnet-v1.5-densepose        | [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)             | 512x512 |

Note that you need download the model and config the model path in [config](./configs/diffusion/diffusion-3d-controlnet.py).
```
coarse_model_and_render = dict(
    diffusion='controlnet',  
    diffusion_path = '/your/path/to/control_dense_pose.ckpt',
    ...
```


## Quickstart
set the text prompt in [config](./configs/diffusion/diffusion-3d-controlnet.py):
```
data = dict(
    ...
    text='Elsa in Frozen Disney',  
```
and run:
``` bash
python3 main.py --config configs/diffusion/diffusion-3d-controlnet.py --render_test --export_mesh --i_print=100 --i_weights=500 --mixed_precision fp16
```
### GPU Memory
We set different configurations for variable GPU memory, and the best results in our paper requirements large GPU memory(~70GB), you can also use less GPU memory(~20GB)  or the default configuration(~32GB) for avatar generation without much degradation in quality.

The configuration of our model is in [config](./configs/diffusion/diffusion-3d-controlnet.py). And the gpu memory requirement can be adjusted by changing the `coarse_model_and_render`, `diffusion_config` and `batch size` in the config file:

Different configuration are listed as following, you can choose one from the `[Low, Medium, High]` list for each parameter, and the GPU memory required are ~20G, ~32G, ~70G respectively.:

```
coarse_model_and_render = dict(
    ...
    num_voxels=[120/120/160]**3,           # expected number of voxel
    num_voxels_base=[120/120/160]**3,      # to rescale delta distance
    ...
    rgbnet_dim=[6/6/12],                 # feature voxel grid dim
    rgbnet_depth=3,               # depth of the colors MLP 
    rgbnet_width=[64/64/128],             # width of the colors MLP
    )

diffusion_config = dict(
    ...
    controlnet=dict(
        height=[256/384/512],
        width=[256/384/512]
    )
)
coarse_train = dict(
    ...
    N_img=[2/3/4],                   # coarse stage batch size 
  )
fine_train = dict(
    dict(
    ...
    N_img=[2/4/8]
```

The generated results are shown below with different configuration. Only some local details are lost, such as the geometric details of the hair on the top of the head, or the texture on the dress:
![configuration](figs/gpu_config.gif)

### Partial Avatars
With our densepose guidance, we can generate partial avatars with only few parts guidance, now we support `full`, `shoulder` and `head` mode, this can be changed in configuration file [config](configs/diffusion/diffusion-3d-controlnet.py):
```
coarse_model_and_render = dict(
    ...
    avatar_type='head',  # full shoulder head
```
![partal](figs/partial.gif)

## Acknowledgement
This repository is heavily based on [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO), [Controlnet](https://github.com/lllyasviel/ControlNet-v1-1-nightly). We would like to thank the authors of these work for publicly releasing their code.

## Citation
``` bibtex
@article{zhang2023avatarverse,
  title={Avatarverse: High-quality \& stable 3d avatar creation from text and pose},
  author={Zhang, Huichao and Chen, Bowen and Yang, Hao and Qu, Liao and Wang, Xu and Chen, Li and Long, Chao and Zhu, Feida and Du, Kang and Zheng, Min},
  journal={arXiv preprint arXiv:2308.03610},
  year={2023}
}
```
