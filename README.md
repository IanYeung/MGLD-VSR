# **Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-resolution**

[![Arxiv](https://img.shields.io/badge/arXiv-2312.00853-b31b1b.svg)](https://arxiv.org/abs/2312.00853)

[Xi Yang](https://scholar.google.com.hk/citations?user=iadRvCcAAAAJ&hl=zh-CN)<sup>1,2</sup> , [Chenhang He](https://skyhehe123.github.io/)<sup>1</sup> , [Jianqi Ma](https://scholar.google.com/citations?user=kQUJjQQAAAAJ&hl=en)<sup>1,2</sup> , [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute

## Abstract
Real-world low-resolution (LR) videos have diverse and complex degradations, imposing great challenges on video super-resolution (VSR) algorithms to reproduce their high-resolution (HR) counterparts with high quality. Recently, the diffusion models have shown compelling performance in generating realistic details for image restoration tasks. However, the diffusion process has randomness, making it hard to control the contents of restored images. This issue becomes more serious when applying diffusion models to VSR tasks because temporal consistency is crucial to the perceptual quality of videos. In this paper, we propose an effective real-world VSR algorithm by leveraging the strength of pre-trained latent diffusion models. To ensure the content consistency among adjacent frames, we exploit the temporal dynamics in LR videos to guide the diffusion process by optimizing the latent sampling path with a motion-guided loss, ensuring that the generated HR video maintains a coherent and continuous visual flow. To further mitigate the discontinuity of generated details, we insert temporal module to the decoder and fine-tune it with an innovative sequence-oriented loss. The proposed motion-guided latent diffusion (MGLD) based VSR algorithm achieves significantly better perceptual quality than state-of-the-arts on real-world VSR benchmark datasets, validating the effectiveness of the proposed model design and training strategies.

## Framework Overview
![mgld](assets/framework-overview.png)

## Updates
- **2023.12.06**: Repo is released.

## Results
<details>
<summary><strong>Synthetic Results</strong> (click to expand) </summary>

![mgld](assets/compare-synthetic.png)

</details>

<details>
<summary><strong>Real-world Results</strong> (click to expand) </summary>

![mgld](assets/compare-real.png)

</details>

<details>
<summary><strong>Sequence Comparison</strong> (click to expand) </summary>
  
![mgld](assets/sequence-compare-020.png)

![mgld](assets/sequence-compare-033.png)

![mgld](assets/sequence-compare-042.png)
</details>

## Miscs

### License
This project is released under the [Apache 2.0 license](LICENSE).

### Citations
```
@article{yang2023mgldvsr,
  title={Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-resolution},
  author={Yang, Xi and He, Chenhang and Ma, Jianqi and Zhang, Lei},
  journal={arXiv preprint arXiv:2312.00853},
  year={2023}
}
```

### Acknowledgement
This implementation largely depends on [StableSR](https://github.com/IceClear/StableSR). We thank the authors for the contribution.