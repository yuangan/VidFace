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
