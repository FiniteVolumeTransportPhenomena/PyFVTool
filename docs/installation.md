# Installation

## Requirements

PyFVTool requires **Python 3.12** and the following packages:

- `numpy` ≥ 2.0
- `scipy`
- `matplotlib`

## Installing with pip

The recommended way to install PyFVTool is directly from GitHub using `pip`:

```bash
pip install git+https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
```

### In a conda environment (recommended)

We strongly recommend creating a dedicated conda environment:

```bash
conda create --name pyfvtool_user python=3.12 numpy scipy matplotlib spyder jupyterlab tqdm
conda activate pyfvtool_user
pip install git+https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
```

Remember to run `conda activate pyfvtool_user` each time before using PyFVTool.

### In Google Colab

Add the following to the first cell of your notebook:

```
!pip install git+https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
```

## Development installation

If you want to modify the source code, install a development (editable) version.
See [](contributing) for details.
