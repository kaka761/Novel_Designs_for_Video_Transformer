# Novel Design Ideas that Improve Video-Understanding Networks with Transformers
This repo is for:

**Paper 1:** ['Novel Design Ideas that Improve Video-Understanding Networks with Transformers'](https://ieeexplore.ieee.org/document/10649969) by Yaxin Hu & Erhardt Barth

&

**Paper 2:** 'How to Efficiently Use Color and Temporal Information for Video Understanding' by Yaxin Hu & Erhardt Barth

## Introduction

**Paper 1:** With the development of deep learning, video understanding has become a promising and challenging research field. In recent years, different transformer architectures have shown state-of-the-art performance on most benchmarks. Although transformers can process longer temporal sequences and therefor perform better than convolution networks, they require huge datasets and have high computational costs. The inputs to video transformers are usually clips sampled out of a video, and the length of the clips is limited by the available computing resources. In this paper, we introduce novel methods to sample and tokenize the input video, such as to better capture the dynamics of the input without a large increase in computational costs. Moreover, we introduce the MinBlocks as a novel architecture inspired by neural processing in biological vision. The combination of variable tubes and MinBlocks improves network performance by 10.67%.

### Novel Designs

**RGBt Sampling**

<div align=left>
<img src="https://github.com/kaka761/Novel_Designs_for_Video_Transformer/blob/master/RGBt.png" align="center" width=45% />
</div>


**Variable Tubes Tokenization**

<div align=left>
<img src="https://github.com/kaka761/Novel_Designs_for_Video_Transformer/blob/master/Tubes.png" align="center" width=45% />
</div>


**MinBlocks**

<div align=left>
<img src="https://github.com/kaka761/Novel_Designs_for_Video_Transformer/blob/master/Mins.png" align="center" width=45% />
</div>

**Paper 2:** The modeling of temporal dependencies, and the associated computational load, remain challenges in video understanding. We here focus on using a more efficient sampling of color and temporal information. We sample color not from the same frame but from different consecutive frames to capture richer temporal information without increasing the computational load. We demonstrate the effectiveness of our approach for 2D-CNNs, 3D-CNNs, and Transformers, for which we obtain significant performance improvements on two benchmarks. The improvements are 2.43% on UCF101 and 4.55% on HMDB51 for the ResNet18, 10.28% and 7.12% for the 3D-ResNet18, and 15.11% and 13.71% for the UniFormerV2. These improvements are obtained without additional costs by just changing the way color is sampled. 

### Architecture

**3DCNN & Transformer**

<div align=left>
<img src="https://github.com/kaka761/Novel_Designs_for_Video_Transformer/blob/master/3Dcnn.png" align="center" width=45% />
</div>

**Fusion Model**

<div align=left>
<img src="https://github.com/kaka761/Novel_Designs_for_Video_Transformer/blob/master/fuse.png" align="center" width=22% />
</div>

## Cite
<details>
<summary>BibTeX entry for citation.</summary>
<pre>
@article{example,
  title={An Example Article},
  author={Doe, John},
  journal={Journal of Example Studies},
  year={2020},
  volume={10},
  number={5},
  pages={123-456},
  doi={10.1234/example}
}
</pre>
</details>

## Acknowledgement
This repository is built based on [UniFormerV2](https://github.com/OpenGVLab/UniFormerV2?tab=readme-ov-file#uniformerv2) repository.
