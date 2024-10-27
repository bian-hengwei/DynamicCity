<img src="assets/logo.png" width="25%" align="left">

# DynamicCity: Large-Scale LiDAR Generation from Dynamic Scenes

![project](https://img.shields.io/badge/Project-%F0%9F%94%97-red)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=dynamic-city.DynamicCity)

![teaser](assets/teaser.webp)

---

## TL;DR
DynamicCity is a cutting-edge 4D LiDAR generation framework designed to capture the dynamic nature of real-world driving environments. It leverages a VAE model with a Projection Module and Expansion & Squeeze Strategy to efficiently represent and reconstruct large-scale LiDAR scenes, achieving up to 12.56 mIoU gain in fitting quality and 2.06x speedup in training. Additionally, a DiT-based diffusion model with a Padded Rollout Operation enables versatile applications like trajectory-driven, inpainting, and layout-conditioned generation. Extensive tests on CarlaSC and Waymo datasets show that DynamicCity significantly surpasses existing 4D LiDAR methods.

---

## News

- **October 2024**: Project page released.

---

## Dynamic Scene Generation Results

### Unconditional Generation
<div class="video-row">
  <img src="assets/u_c_1.webp" alt="Unconditional Generation 1" width="240">
  <img src="assets/u_c_2.webp" alt="Unconditional Generation 2" width="240">
  <img src="assets/u_c_3.webp" alt="Unconditional Generation 3" width="240">
</div>

### HexPlane Conditional Generation
<div class="video-row">
  <img src="assets/h_c_1.webp" alt="HexPlane Conditional Generation 1" width="240">
  <img src="assets/h_c_2.webp" alt="HexPlane Conditional Generation 2" width="240">
  <img src="assets/h_c_3.webp" alt="HexPlane Conditional Generation 3" width="240">
</div>

### Command/Trajectory-Driven Generation
<div class="video-row">
  <img src="assets/r_c_1.webp" alt="Command-Driven Generation 1" width="240">
  <img src="assets/r_c_2.webp" alt="Command-Driven Generation 2" width="240">
  <img src="assets/r_c_3.webp" alt="Command-Driven Generation 3" width="240">
</div>

### Layout-Conditioned Generation
<div class="video-row">
  <img src="assets/l_c_1.webp" alt="Layout-Conditioned Generation 1" width="360">
  <img src="assets/l_c_2.webp" alt="Layout-Conditioned Generation 2" width="360">
</div>

### Dynamic Scene Inpainting
<div class="video-row">
  <img src="assets/i_c_1.webp" alt="Dynamic Scene Inpainting 1" width="360">
  <img src="assets/i_c_2.webp" alt="Dynamic Scene Inpainting 2" width="360">
</div>

## TODO List

- [ ] Release code
- [ ] Release model weights and pretrained checkpoints
