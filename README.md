# uResNet_unofficial
Unofficial reimplementation of the paper:

Guerrero, Ricardo, et al. "White matter hyperintensity and stroke lesion segmentation and differentiation using convolutional neural networks." NeuroImage: Clinical 17 (2018): 918-934. https://doi.org/10.1016/j.nicl.2017.12.022


Used in:

Phitidis, Jesse, et al. "Segmentation of White Matter Hyperintensities and Ischaemic Stroke Lesions in Structural MRI." Annual Conference on Medical Image Understanding and Analysis. Cham: Springer Nature Switzerland, 2023. https://doi.org/10.1007/978-3-031-48593-0_1

## Usage

Create an environment with python 3.9.13 and then:

```bash
pip install -r requirements.txt
```

Refer to the pytorch lightning cli documentation and run training with:

```bash
python runner.py fit --config path_to_config
```
