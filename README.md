# VidFace
(The official code)
（coming soon）

### :book: VidFace: A Full-Transformer Solver for Video FaceHallucination with Unaligned Tiny Snapshots
> [[Paper](https://arxiv.org/abs/2105.14954)]

#### Abstract

In this paper, we investigate the task of hallucinating an authentic high-resolution (HR) human face from multiple low-resolution (LR) video snapshots. We propose a pure transformer-based model, dubbed VidFace, to fully exploit the full-range spatio-temporal information and facial structure cues among multiple thumbnails. Specifically, VidFace handles multiple snapshots all at once and harnesses the spatial and temporal information integrally to explore face alignments across all the frames, thus avoiding accumulating alignment errors. Moreover, we design a recurrent position embedding module to equip our transformer with facial priors, which not only effectively regularises the alignment mechanism but also supplants notorious pre-training. Finally, we curate a new large-scale video face hallucination dataset from the public Voxceleb2 benchmark, which challenges prior arts on tackling unaligned and tiny face snapshots. To the best of our knowledge, we are the first attempt to develop a unified transformer-based solver tailored for video-based face hallucination. Extensive experiments on public video face benchmarks show that the proposed method significantly outperforms the state of the arts.

#### BibTeX
    @article{gan2021vidface,
      title={VidFace: A Full-Transformer Solver for Video FaceHallucination with Unaligned Tiny Snapshots},
      author={Gan, Yuan and Luo, Yawei and Yu, Xin and Zhang, Bang and Yang, Yi},
      journal={arXiv preprint arXiv:2105.14954},
      year={2021}
    }
    
## :wrench: Dependencies and Installation
(This work is based on the framework of [BasicSR](https://github.com/xinntao/EDVR))
- Python >= 3.7
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
1. Clone repo

    ```bash
    git clone https://github.com/yuangan/VidFace.git
    ```

1. Install dependent packages

    ```bash
    cd VidFace
    pip install -r requirements.txt
    ```

1. Install VidFace
    ```
    python setup.py develop
    ```
    
    You may also want to specify the CUDA paths:

      ```bash
      CUDA_HOME=/usr/local/cuda \
      CUDNN_INCLUDE_DIR=/usr/local/cuda \
      CUDNN_LIB_DIR=/usr/local/cuda \
      python setup.py develop
      ```
      
VidFace has been tested on Linux and Windows with anaconda.

## :package: Dataset Preparation
1. TUFS145K images can be downloaded from [Google](https://drive.google.com/drive/folders/1xsq09wmr8iXxhHBtsseY4Iwq6oH2i9mW?usp=sharing) or [Baidu](), then excute ```cat tufs145ka* > tufs145k.zip``` and extract it to VidFace fold.
3. TUFS145K landmarks can be downloaded from [Google](https://drive.google.com/drive/folders/1xsq09wmr8iXxhHBtsseY4Iwq6oH2i9mW?usp=sharing) or [Baidu](), extract it to './landmarks'

1. Prepare your dataset
- Please refer to **[DatasetPreparation.md](docs/DatasetPreparation.md)** for more details.

## :computer: Train and Test

- **Training and testing commands**: 


