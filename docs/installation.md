# Installation

## Requirements

PyFVTool requires Python 3.12 and the following packages:

- `numpy` ≥ 2.0
- `scipy`
- `matplotlib`

## Installing with pip

PyFVTool can be simply installed from the command line using `pip`.

```
pip install pyfvtool
```

## Python environment

PyFVTool is best used with Python 3.12 and the most recent NumPy and SciPy versions. We recommend the [conda-forge](https://conda-forge.org/download/) Python distribution. A specific environment for PyFVTool may be set up as follows.

```
conda create --name pyfvtool_user python=3.12 numpy scipy matplotlib spyder jupyterlab tqdm

conda activate pyfvtool_user
```

Of course, do not forget to  `conda activate pyfvtool_user`  the environment every time you run Python code that uses PyFVTool.

## JupyterLite, Google Colab

PyFVTool can be used inside [JupyterLite](https://jupyter.org/try-jupyter). Your calculations will be run completely inside your web browser! No local Python installation needed. To install PyFVTool in JupyterLite, enter the following in the first cell:

```python
import piplite
await piplite.install("pyfvtool")
```

PyFVTool also works in [Google Colab](https://colab.research.google.com/), you can enter the following in the first cell of a Colab Notebook to install it in the current Colab instance:

```
!pip install pyfvtool
```


## Development installation
If you would like to work on the source code, it is possible to install a development version. See ["Contributing"](./contributing.md).
