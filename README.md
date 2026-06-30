Code: https://github.com/wangrui2025/gdkvm_code
# GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule

[![Project Page](https://img.shields.io/badge/Project-Page-BC52EE?logo=githubpages)](https://wangrui2025.github.io/GDKVM/)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv2025.thecvf.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Rui Wang, Yimu Sun, Jingxing Guo, Huisi Wu, Jing Qin**
>
> College of Computer Science and Software Engineering, Shenzhen University
>
> Centre for Smart Health, School of Nursing, The Hong Kong Polytechnic University

## Abstract

Accurate segmentation of cardiac chambers in echocardiography sequences is crucial for the quantitative analysis of cardiac function, aiding in clinical diagnosis and treatment. The imaging noise, artifacts, and the deformation and motion of the heart pose challenges to segmentation algorithms.

While existing methods based on convolutional neural networks, Transformers and space-time memory networks, have improved segmentation accuracy, they often struggle with the trade-off between capturing long-range spatiotemporal dependencies and maintaining computational efficiency with fine-grained feature representation.

In this paper, we introduce **GDKVM**, a novel architecture for echocardiography video segmentation. The model employs **Linear Key-Value Association (LKVA)** to effectively model inter-frame correlations, and introduces **Gated Delta Rule (GDR)** to efficiently store intermediate memory states. **Key-Pixel Feature Fusion (KPFF)** module is designed to integrate local and global features at multiple scales, enhancing robustness against boundary blurring and noise interference.

We validated GDKVM on two mainstream echocardiography video datasets (**CAMUS** and **EchoNet-Dynamic**) and compared it with various state-of-the-art methods. Experimental results show that GDKVM outperforms existing approaches in terms of segmentation accuracy and robustness, while ensuring real-time performance.

## Project Page

[https://wangrui2025.github.io/GDKVM/](https://wangrui2025.github.io/GDKVM/)

## Installation

```bash
git clone https://github.com/wangrui2025/GDKVM.git
cd GDKVM
pip install -e .
```

## Quick Start

TBD — code is being organized for release.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{wang2025gdkvm,
  title={GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule},
  author={Wang, Rui and Sun, Yimu and Guo, Jingxing and Wu, Huisi and Qin, Jing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## License

This project is released under the MIT License.
