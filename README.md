## Neural Graph Differential Equations (Neural GDEs)

> We introduce the framework of continuous-depth graph neural networks (GNNs). Graph neural ordinary differential equations (GDEs) are formalized as the counterpart to GNNs where the input-output relationship is determined by a continuum of GNN layers, blending discrete topological structures and differential equations. The proposed framework is shown to be compatible with various static and autoregressive GNN models. Results prove general effectiveness of GDEs: in static settings they offer computational advantages by incorporating numerical methods in their forward pass; in dynamic settings, on the other hand, they are shown to improve performance by exploiting the geometry of the underlying dynamics.


paper: [arXiv link](https://arxiv.org/abs/1911.07532)

<p align="center"> 
<img src="fig/gde_vec.jpg" width="450" height="300">
</p>

This repository contains examples of *Neural Graph Differential Equations* (**Neural GDE**). The tutorial notebook contains abundant amounts of comments and all runnable top-to-bottom.

Neural GDEs rely on [dgl](https://github.com/dmlc/dgl) and [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

NOTE: Neural GDE model zoo and additional tutorials are included in the `torchdyn` library: [link](https://github.com/DiffEqML/torchdyn)

If you find our work useful, consider citing us:

```
@article{poli2019graph,
  title={Graph Neural Ordinary Differential Equations},
  author={Poli, Michael and Massaroli, Stefano and Park, Junyoung and Yamashita, Atsushi and Asama, Hajime and Park, Jinkyoo},
  journal={arXiv preprint arXiv:1911.07532},
  year={2019}
}
```
