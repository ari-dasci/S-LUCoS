# LUCoS: Latent Unsupervised Context Selection for Tabular Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.27254-b31b1b.svg)](https://arxiv.org/abs/2605.27254)

LUCoS is a label-free context selection framework for Tabular Foundation Models (TFMs) operating in low-label regimes.

The method studies the cold-start setting, where instances must be selected for annotation before any labels are available. LUCoS replaces unreliable raw-feature geometry with latent representations induced by unsupervised Prior-Fitted Networks (PFNs), enabling simple geometric selection methods such as K-Medoids to identify informative context subsets.

The selected instances are then annotated and used as the in-context support set for downstream prediction with supervised TFMs such as TabPFN.

## Main idea

Our experiments on 67 OpenML-CC18 datasets show that:
- LUCoS consistently ranks first across multiple low-label budgets.
- Representation quality becomes the dominant factor as the labeling budget increases.
- Original-space geometric selection frequently collapses below random selection, while latent-space selection remains robust.

## Repository status

⚠️ The repository is currently being cleaned and prepared for public release.

Code, experimental scripts, and reproduction instructions will be uploaded in the next few days.

## Paper

**LUCoS: Latent Unsupervised Context Selection for Tabular Foundation Models**

[https://arxiv.org/abs/2605.27254](https://arxiv.org/abs/2605.27254)

## Citation

If you use this work, please cite:

```bibtex
@misc{ipas2026lucos,
      title={LUCoS: Latent Unsupervised Context Selection for Tabular Foundation Models}, 
      author={Oroel Ipas and Guillermo Gomez-Trenado and Rocío Romero-Zaliz and Isaac Triguero},
      year={2026},
      eprint={2605.27254},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.27254}, 
}
```

