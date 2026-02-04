<h1 align="center"><strong>ConsistentRFT: Reducing Visual Hallucinations in Flow-based Reinforcement Fine-Tuning</strong></h1>
<p align="center">
  <a href="https://xiaofeng-tan.github.io/" target="_blank">Xiaofeng Tan<sup>1,3,†,*</sup></a>&emsp;
  Jun Liu<sup>3,†</sup>&emsp;
  Yuanting Fan<sup>3</sup>&emsp;
  Bin-Bin Gao<sup>3</sup>&emsp;
  Xi Jiang<sup>2</sup>&emsp;
  Xiaochen Chen<sup>3</sup>&emsp;
  Jinlong Peng<sup>3</sup>&emsp;
  Chengjie Wang<sup>3</sup>&emsp;
  Hongsong Wang<sup>1,‡</sup>&emsp;
  Feng Zheng<sup>2,‡</sup>
  <br>
  <sup>1</sup>Southeast University&emsp;
  <sup>2</sup>Southern University of Science and Technology&emsp;
  <sup>3</sup>Tencent Youtu Lab
  <br>
  <sup>†</sup>Equal contribution.&emsp;
  <sup>‡</sup>Corresponding authors.&emsp;
  <sup>*</sup>Work done during Xiaofeng Tan's internship at Tencent Youtu Lab.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.03425">
    <img src="https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow" alt="Paper PDF on arXiv">
  </a>
  <a href="https://xiaofeng-tan.github.io/projects/ConsistentRFT/">
    <img src="https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green" alt="Project Page">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black" alt="HuggingFace Models">
  </a>
</p>

<p align="left">
  <span style="color: #84193E;"><b>Why</b></span> do visual hallucinations arise in reinforcement fine-tuning, and <span style="color: #004D99;"><b>how</b></span> can we reduce them?
</p>

<blockquote>
<p><b>TL;DR:</b> This work provides a preliminary analysis of this issue from two perspectives—<span style="color: #84193E;"><b>limited exploration</b></span> and <span style="color: #84193E;"><b>trajectory imitation</b></span>—and proposes addressing them using <span style="color: #004D99;"><b>Dynamic Granularity Rollout</b></span> and <span style="color: #004D99;"><b>Consistent Policy Gradient Optimization</b></span>.</p>
</blockquote>

## 📢 NEWS
News: 🚀 The code will be partially released before **February 15th, 2026**, with the full release scheduled before **April 2026**.


We sincerely appreciate your interest in our work. Due to **mandatory internal review** and approval procedures at our institution, the release of the paper and code is currently **pending**. We will make them publicly available immediately **as soon as the process is completed**. Thank you for your patience and understanding. 

## 🙏 Acknowledgement

This work is built on many amazing research works and open-source projects. Thanks to all the authors for sharing!

- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [Flow-GRPO](https://github.com/yifan123/flow_grpo)
- [MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO)
- [DDPO](https://github.com/jannerm/ddpo)
- [Pref-GRPO](https://github.com/CodeGoat24/Pref-GRPO)

## 📝 Citation

If you find this repository helpful in your research, please consider citing the paper and starring the repo ⭐.

```bibtex
@article{tan2026consistentrft,
  title={ConsistentRFT: Reducing Visual Hallucinations in Flow-based Reinforcement Fine-Tuning},
  author={Tan, Xiaofeng and Liu, Jun and Fan, Yuanting and Gao, Bin-Bin and Jiang, Xi and Chen, Xiaochen and Peng, Jinlong and Wang, Chengjie and Wang, Hongsong and Zheng, Feng},
  journal={arXiv preprint arXiv:2602.03425},
  year={2026}
}
```
