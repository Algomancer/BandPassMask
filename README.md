# Bandmask

A simple masking strategy that seems to outperform random and block masking. Tune sigma for appropriate local and global structure, or randomize them for training on different masking distributions. 

After digging around, seems similar to the 'green mask' in ColorMAE. Supports subset targets.

```
@misc{algomancer2025,
  author = {@algomancer},
  title  = {Some Dumb Shit},
  year   = {2025}
}
```

```
@misc{hinojosa2024colormaeexploringdataindependentmasking,
      title={ColorMAE: Exploring data-independent masking strategies in Masked AutoEncoders}, 
      author={Carlos Hinojosa and Shuming Liu and Bernard Ghanem},
      year={2024},
      eprint={2407.13036},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.13036}, 
}
```
