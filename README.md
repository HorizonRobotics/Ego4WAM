<div align="center">
<p><strong><font size="5">Ego4WAM: What Matters When Scaling Egocentric Human Data for Robot Learning?</font></strong></p>

<a href="https://sunzhihao18.github.io/Ego4WAM/"><img src="https://img.shields.io/badge/Project-Page-7c3aed?style=for-the-badge&logo=githubpages" alt="Project Page"></a>
<a href="https://huggingface.co/HorizonRobotics/Ego4WAM"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow?style=for-the-badge&logo=huggingface" alt="Model on Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"></a>
</div>

## 📖 Abstract

Egocentric human data provides a scalable source of experience for robot learning, but varies substantially in human-robot alignment, behavioral coverage, and available supervision. Existing work shows favorable scaling with increasing human data, but it remains unclear which data properties drive downstream robot gains and how to use such data throughout the training pipeline. We present a systematic study of egocentric human data with different alignment and supervision under a unified world-action model framework. With the model backbone fixed, we disentangle the effects of human-robot alignment, data duration and task diversity, action supervision, and data usage strategies. We find that aligned human demonstrations substantially improve out-of-distribution generalization and reduce target-task robot data requirements; data duration and task diversity affect downstream capabilities differently; and video-only supervision remains effective without action labels, providing a strong foundation for subsequent video-action training. We validate these findings through closed-loop policy evaluation on both real robots and RoboDojo. Rather than treating data duration as the sole scaling axis, Ego4WAM shows how alignment, task diversity, available supervision, and usage strategy jointly shape the value of egocentric human data for robot learning.

## ⭐ Key Findings

1. **Robot-aligned human data enables OOD adaptation and reduces robot data.**
   * **OOD Adaptation.** Aligned human demonstrations expose the policy to deployment conditions absent from robot training. Without any object- or scene-OOD robot demonstrations, object-OOD success increases from 10% to 60%, while scene-OOD success increases from 0% to 20% with a substantially larger gain in task progress.
   * **Reducing Robot-Data Dependence.** With 100 robot demonstrations, aligned human data recovers 80% ID, 60% object-OOD, and 80% scene-OOD success. Human data alone is insufficient when robot supervision falls to 20 demonstrations, but broad multi-task mid-training makes the same limited downstream data effective, reaching 80%, 80%, and 60% success, respectively.

2. **Scaling egocentric data does not always lead to better performance.**
   **Data duration** and **task diversity** are distinct scaling axes, and total hours alone do not characterize useful scale. Denser coverage of recurring tasks provides limited additional gains once common behaviors are sufficiently represented, while expanding aggressively into increasingly sparse long-tail tasks can introduce negative transfer and hurt precision-sensitive performance.

3. **Egocentric data provides a strong dynamics prior for world-action learning.**
   * **Learning Dynamics Priors without Action Supervision.** Video-only pre-training improves the average RoboDojo score from 6.39 to 14.13 and success rate from 3.15% to 9.45% without human action labels, extending usable data beyond the action-valid subset. The learned dynamics prior further supports video-action training, while joint future conditioning exploits it most effectively, reaching a 20.29 average score and 14.35% success rate.
   * **Competitive Performance with Limited Robot Data.** With only **approximately 90 hours of real-robot data** and **15K hours of egocentric data**, Ego4WAM achieves a competitive 20.29 average score and 14.35% success rate on RoboDojo. Because the model is not explicitly designed for long-term memory or VLM-based open-instruction following, Memory and Open remain its weaker categories; nevertheless, Ego4WAM achieves the best Precision score and success rate and the best Long-Horizon score among the reported methods.

## 📚 Citation

If you find Ego4WAM useful in your research, please cite:

```bibtex
@article{sun2026ego4wam,
  title   = {Ego4WAM: What Matters When Scaling Egocentric Human Data for Robot Learning?},
  author  = {Sun, Zhihao and Liu, Liu and Wang, Xinjiang and Jiang, Haoyi and Feng, Wei and Zhang, Huiqiang and Jia, Xiaosong and Su, Zhizhong and Wu, Zuxuan},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## 🙏 Acknowledgements

Ego4WAM builds on several open-source projects and research efforts:

* **[StarVLA](https://github.com/starVLA/starVLA)** provides the modular training, data, and deployment infrastructure used by this repository.
* **[FastWAM](https://arxiv.org/abs/2603.16666)** provides the world-action modeling foundation on which the Ego4WAM architecture is built.
* **[Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)** provides the video world-model backbone.
* **[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)** provides the standardized manipulation benchmark and evaluation protocol, with **[XPolicyLab](https://github.com/JinhuiYE/XPolicyLab)** supporting policy integration and evaluation.
* **[LeRobot](https://github.com/huggingface/lerobot)** provides dataset formats and data-loading components used in the training pipeline.

We thank the authors and maintainers of these projects, as well as the creators of the public egocentric datasets used in this study.

## 📄 License

This repository is released under the [MIT License](LICENSE). Portions of the code are derived from StarVLA and FastWAM under their respective licenses. Third-party models, datasets, benchmarks, and assets remain subject to their original licenses and terms of use.
