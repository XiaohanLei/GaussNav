# GaussNav

PyTorch implementation of paper: GaussNav: Gaussian Splatting for Visual Navigation

[Project Page](https://xiaohanlei.github.io/projects/GaussNav/)<br />

### Overview:

Our GaussNav framework consists of three stages, including Frontier Exploration, Semantic Gaussian Construction and Gaussian Navigation. First, the agent employs Frontier Exploration to collect observations of the unknown environment. Second, the collected observations are used to construct Semantic Gaussian. By leveraging semantic segmentation algorithms, we assign semantic labels to each Gaussian. We then cluster Gaussians with their semantic labels and 3D positions, segmenting objects in the scene into different instances under various semantic categories. This representation is capable of preserving not only the 3D geometry of the scene and the semantic labels of each Gaussian, but also the texture details of the scene, thereby enabling novel view synthesis. Third, we render descriptive images for object instances, matching them with the goal image to effectively locate the target object. Upon determining the predicted goal object’s position, we can efficiently transform our Semantic Gaussian into grid map and employ path planning algorithms to accomplish the navigation.

### This repository contains:

The Gaussian Navigation module of GaussNav (which can be used for code base for either PointGoal task or the upper bound of planning-based methods of Instance ImageNav task).
#### We will soon release the code of the Semantic Gaussian Construction module and previously constructed 3D model.

## Installing Dependencies
- We use v0.2.3 of [habitat-sim](https://github.com/facebookresearch/habitat-sim), please follow the instructions to complete installation
- Install habitat-lab:
```
cd GaussianNavigation\3rdparty\habitat-lab-0.2.3
pip install -e habitat-lab
```
- Install [pytorch](https://pytorch.org/) according to your system configuration
- cd to the root directory, install requirements
```
pip install -r requirements.txt
```

### Downloading scene dataset and episode dataset
- Follow the instructions in [habitat-lab](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)
- Move the dataset or create a symlink at `GaussianNavigation/data`

## Test setup
To verify that the data is setup correctly, run:
```
cd GaussianNavigation
python run.py
```

## Some tips
- If you want to use our code base to build your own project, you can write your own envs in `GaussianNavigation\vector_env\envs`
- If you have any questions, feel free to open issues




## TODO list:
- [x] release the code of Gaussian Navigation
- [ ] release the code of Semantic Gaussian Construction
- [ ] release the model of previously constructed Semantic Gaussian


This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.

### Bibtex:
```
@misc{lei2024gaussnavgaussiansplattingvisual,
      title={GaussNav: Gaussian Splatting for Visual Navigation}, 
      author={Xiaohan Lei and Min Wang and Wengang Zhou and Houqiang Li},
      year={2024},
      eprint={2403.11625},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.11625}, 
}
```
