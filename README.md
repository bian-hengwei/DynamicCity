<img src="https://dynamic-city.github.io/assets/images/logo.png" width="12.5%" align="left">

# DynamicCity: Large-Scale LiDAR Generation from Dynamic Scenes

<p align="center">
  <a href="https://dynamic-city.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-red">
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=dynamic-city.DynamicCity">
  </a>
</p>

<img src="https://dynamic-city.github.io/assets/images/teaser_small.webp" alt="Teaser" width="100%">

LiDAR scene generation has been developing rapidly recently. However, existing methods primarily focus on generating static and single-frame scenes, overlooking the inherently dynamic nature of real-world driving environments. In this work, we introduce **DynamicCity**, a novel 4D LiDAR generation framework capable of generating large-scale, high-quality LiDAR scenes that capture the temporal evolution of dynamic environments. DynamicCity mainly consists of two key models: 1. A VAE model for learning HexPlane as the compact 4D representation. Instead of using naive averaging operations, DynamicCity employs a novel **Projection Module** to effectively compress 4D LiDAR features into six 2D feature maps for HexPlane construction, which significantly enhances HexPlane fitting quality (up to **12.56** mIoU gain). Furthermore, we utilize an **Expansion & Squeeze Strategy** to reconstruct 3D feature volumes in parallel, which improves both network training efficiency and reconstruction accuracy than naively querying each 3D point (up to **7.05** mIoU gain, **2.06x** training speedup, and **70.84%** memory reduction). 2. A DiT-based diffusion model for HexPlane generation. To make HexPlane feasible for DiT generation, a **Padded Rollout Operation** is proposed to reorganize all six feature planes of the HexPlane as a squared 2D feature map. In particular, various conditions could be introduced in the diffusion or sampling process, supporting **versatile 4D generation applications**, such as trajectory- and command-driven generation, inpainting, and layout-conditioned generation. Extensive experiments on the CarlaSC and Waymo datasets demonstrate that DynamicCity significantly outperforms existing state-of-the-art 4D LiDAR generation methods across multiple metrics. The code will be released to facilitate future research.

# Overview
<img src="https://dynamic-city.github.io/assets/images/pipeline.png" alt="Overview" width="100%">

Our **DynamicCity** framework consists of two key procedures: **(a)** Encoding HexPlane with an VAE architecture, and **(b)** 4D Scene Generation with HexPlane DiT.

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
| <img src="https://dynamic-city.github.io/assets/images/u_c_1.webp" alt="Unconditional Generation 1" width="240">|<img src="https://dynamic-city.github.io/assets/images/u_c_2.webp" alt="Unconditional Generation 2" width="240">|<img src="https://dynamic-city.github.io/assets/images/u_c_3.webp" alt="Unconditional Generation 3" width="240">|
||||

### HexPlane Conditional Generation
||||
|-|-|-|
| <img src="https://dynamic-city.github.io/assets/images/h_c_1.webp" alt="Unconditional Generation 1" width="240">|<img src="https://dynamic-city.github.io/assets/images/h_c_2.webp" alt="Unconditional Generation 2" width="240">|<img src="https://dynamic-city.github.io/assets/images/h_c_3.webp" alt="Unconditional Generation 3" width="240">|
||||


### Command & Trajectory-Driven Generation
||||
|-|-|-|
| <img src="https://dynamic-city.github.io/assets/images/r_c_1.webp" alt="Unconditional Generation 1" width="240">|<img src="https://dynamic-city.github.io/assets/images/r_c_2.webp" alt="Unconditional Generation 2" width="240">|<img src="https://dynamic-city.github.io/assets/images/r_c_3.webp" alt="Unconditional Generation 3" width="240">|
||||


### Layout-Conditioned Generation
|||
|-|-|
| <img src="https://dynamic-city.github.io/assets/images/l_c_1.webp" alt="Unconditional Generation 1" width="360">|<img src="https://dynamic-city.github.io/assets/images/l_c_2.webp" alt="Unconditional Generation 2" width="360">|
|||


### Dynamic Scene Inpainting
|||
|-|-|
| <img src="https://dynamic-city.github.io/assets/images/i_c_1.webp" alt="Unconditional Generation 1" width="360">|<img src="https://dynamic-city.github.io/assets/images/i_c_2.webp" alt="Unconditional Generation 2" width="360">|
|||



## :memo: TODO List
- [ ] Release code
- [ ] Release model weights and pretrained checkpoints



