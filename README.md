# Uniformity First: Uniformity-aware Test-time Adaptation of Vision-language Models against Image Corruption
The official code for "Uniformity First: Uniformity-aware Test-time Adaptation of Vision-language Models against Image Corruption."  
[[arXiv](https://arxiv.org/abs/2505.12912)]

![Overview](overview.png)

## Environment
- Prepare the datasets (ImageNet-C, ImageNet-C-bar) and write their path in `dataset/dataset_config.py`.
- Install dependencies or build the docker image according to `docker/Dockerfile`.

```bash
$ docker build -t tta_uninfo docker --no-cache
```


## TTA
```bash
$ python3 main.py -c imagenet-c.yaml -o result
```


## Citation
If our work assists your research, please cite our paper:

```
@article{adachi2025uninfo,
title={Uniformity First: Uniformity-aware Test-time Adaptation of Vision-language Models against Image Corruption},
author={Kazuki Adachi and Shin'ya Yamaguchi and Tomoki Hamagami},
journal={arXiv preprint arXiv:2505.12912},
year={2025}
}
```
