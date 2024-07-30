# PyFVTool development and testing

## Development environment

For development, `git clone` the repository to a local directory, or unpack the ZIP downloaded from GitHub.

A suitable development and test Python environment can be created with conda:

```
conda create --name pyfvtool_dev python numpy scipy matplotlib pypardiso spyder jupyterlab pytest=7.4 pytest_notebook
conda activate pyfvtool_dev
```

Once the environment configured and activated, you can change your working directory to the local copy of the `pyfvtool` repository and install an editable (development) version.

```
pip install --editable .
```

(Do not forget the trailing dot!)


## Testing

Simple automated testing of the Python code can be done using `pytest` for which a configuration has been set up. The configuration is minimalist. It enables to conveniently test if the PyFVTool still works as expected before committing changes to the code repository. A full 100 percent code testing coverage is not guaranteed at this stage. 

All available tests can be run by simply invoking

```   
pytest
```  

from the command line, when in the `pyfvtool` project root development directory.

Running tests requires to have installed in your Python environment:
- `pytest`
- `pytest_notebook` (**mind the underscore**, *do not use a dash*!)

The latter two can be installed, using `conda install pytest=7.4 pytest_notebook` (**mind the underscore**, *do not use a dash*!). If you do not have `pytest_notebook` and Jupyter Notebook available, it should be possible to run `pytest` nevertheless by removing the lines starting with `nb_` from the `pytest.ini` file. **NOTE: Currently `pytest_notebook` does not work with the latest pytest 8.0. Therefore, pytest has been downgraded to 7.4, awaiting a fix.**

The present pytest configuration for PyFVTool scans all directories for files named `test_*.py` or `*_test.py`, or `*.ipynb` (Notebooks). These will be considered "tests". Several tests compare the finite-volume result to the known analytic result for textbook cases, and thus provide a rudimentary form of functional numerical testing as well.
