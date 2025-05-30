# PyFVTool development and testing

## Development environment

For development, `git clone` the repository to a local directory, or unpack the ZIP downloaded from GitHub.

A suitable development and test Python environment can be created with conda:

```
conda create --name pyfvtool_dev python=3.12 numpy=1.26.4 scipy matplotlib pypardiso spyder jupyterlab pytest tqdm
conda activate pyfvtool_dev
```

**NOTE.** *NumPy 2.x seems to break `pypardiso`. Use NumPy 1.26.4.*

Once the environment configured and activated, you can change your working directory to the local copy of the `pyfvtool` repository and install an editable (development) version.

```
pip install --editable .
```

(Do not forget the trailing dot!)


## Testing

Simple automated testing of the Python code can be done using `pytest`. A minimalist `pytest` configuration has been set up. It enables to conveniently test if the PyFVTool still works as expected before committing changes to the code repository. A full 100 percent code testing coverage is not guaranteed at this stage.

The present pytest configuration for PyFVTool scans all directories for files named `test_*.py` or `*_test.py`. These will be considered "tests". Several tests compare the finite-volume result to the known analytic result for textbook cases, and thus provide a form of functional numerical testing as well.

All available tests can be run by simply invoking

```   
pytest
```  

from the command line, when in the `pyfvtool` project root development directory.

The full test takes several minutes to complete (almost 3 minutes on a typical Windows 11 laptop), since it involves many cycles of actually solving partial differential equations numerically.

The collection of test scripts has not been optimized. Most of the test scripts are modified example scripts, which have been adapted to work with `pytest` by adding simple `assert` statements in test functions.


### Testing environment

Running tests requires to have installed in your Python environment:
- `pytest`

It can be installed, using `conda install pytest`
