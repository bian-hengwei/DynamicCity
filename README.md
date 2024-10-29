<img src="assets/logo.png" width="12%" align="left">

# DynamicCity: Large-Scale LiDAR Generation from Dynamic Scenes

![project](https://img.shields.io/badge/Project-%F0%9F%94%97-red)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=dynamic-city.DynamicCity)

|![teaser](assets/teaser.webp)|
|:-:|
|DynamicCity is a cutting-edge 4D LiDAR generation framework designed to capture the dynamic nature of real-world driving environments. It leverages a VAE model with a Projection Module and Expansion & Squeeze Strategy to efficiently represent and reconstruct large-scale LiDAR scenes, achieving up to 12.56 mIoU gain in fitting quality and 2.06x speedup in training. Additionally, a DiT-based diffusion model with a Padded Rollout Operation enables versatile applications like trajectory-driven, inpainting, and layout-conditioned generation. Extensive tests on CarlaSC and Waymo datasets show that DynamicCity significantly surpasses existing 4D LiDAR methods.|


## Updates

- **[October 2024]**: Project page released.


## Outline

- [Installation](#gear-installation)
- [Data Preparation](#hotsprings-data-preparation)
- [Getting Started](#rocket-getting-started)
- [Dynamic Scene Generation](#cityscape-dynamic-scene-generation)
- [TODO List](#memo-todo-list)
  

## :gear: Installation
Kindly refer to [INSTALL.md](docs/INSTALL.md) for the installation details.


## :hotsprings: Data Preparation
Kindly refer to [DATA_PREPARE.md](docs/INSTALL.md) for the details to prepare the [CarlaSC](), [Occ3D-Waymo](), and [Occ3D-nuScenes]() datasets.


## :rocket: Getting Started
Kindly refer to [GET_STARTED.md](docs/GET_STARTED.md) to learn more about how to use this codebase.



## :cityscape: Dynamic Scene Generation

### Unconditional Generation
||||
|-|-|-|
| <img src="assets/u_c_1.webp" alt="Unconditional Generation 1" width="240">|<img src="assets/u_c_2.webp" alt="Unconditional Generation 2" width="240">|<img src="assets/u_c_3.webp" alt="Unconditional Generation 3" width="240">|
||||

### HexPlane Conditional Generation
||||
|-|-|-|
| <img src="assets/h_c_1.webp" alt="Unconditional Generation 1" width="240">|<img src="assets/h_c_2.webp" alt="Unconditional Generation 2" width="240">|<img src="assets/h_c_3.webp" alt="Unconditional Generation 3" width="240">|
||||


### Command & Trajectory-Driven Generation
||||
|-|-|-|
| <img src="assets/r_c_1.webp" alt="Unconditional Generation 1" width="240">|<img src="assets/r_c_2.webp" alt="Unconditional Generation 2" width="240">|<img src="assets/r_c_3.webp" alt="Unconditional Generation 3" width="240">|
||||


### Layout-Conditioned Generation
|||
|-|-|
| <img src="assets/l_c_1.webp" alt="Unconditional Generation 1" width="360">|<img src="assets/l_c_2.webp" alt="Unconditional Generation 2" width="360">|
|||


### Dynamic Scene Inpainting
|||
|-|-|
| <img src="assets/i_c_1.webp" alt="Unconditional Generation 1" width="360">|<img src="assets/i_c_2.webp" alt="Unconditional Generation 2" width="360">|
|||



## :memo: TODO List
- [ ] Release code
- [ ] Release model weights and pretrained checkpoints



