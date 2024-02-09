# PyFVTool development and testing

## Development environment

For development, `git clone` the repository to a local directory, or unpack the downloaded ZIP.

A suitable development and test Python environment can be created with conda/mamba:

```
conda create --name pyfvtool_dev python numpy scipy matplotlib spyder jupyterlab pytest=7.4 pytest_notebook
conda activate pyfvtool_dev
```

Once the environment configured and activated, you can change your working directory to the local copy of the pyfvtool repository and install an editable (development) version.

```
pip install --editable .
```


## Testing

Simple automated testing of the Python code can be done using `pytest` for which a configuration has been set up. The configuration is minimalist. It enables to conveniently test if the PyFVTool still works as expected before commiting changes to the code repository. A full 100 percent code testing coverage is not guarantueed at this stage. 

All available tests can be run by simpling invoking

```   
pytest
```  

from the command line, when in the `pyfvtool` project root development directory.

Running tests requires to have installed in your Python environment:
- `pytest`
- `pytest_notebook` (**mind the underscore**, *do not use a dash*!)

The latter two can be installed, using `conda install pytest=7.4 pytest_notebook` (**mind the underscore**, *do not use a dash*!). If you do not have `pytest_notebook` and Jupyter Notebook available, it should be possible to run `pytest` nevertheless by removing the lines starting with `nb_` from the `pytest.ini` file. **NOTE: Currently `pytest_notebook` is broken and will not be used in our test configuration, awaiting a fix.**

The present pytest configuration for PyFVTool scans all directories for files named `test_*.py` or `*_test.py`, or `*.ipynb` (Notebooks). These will be considered "tests". Several tests compare the finite-volume result to the known analytic result for textbook cases, and thus provide a rudimentary form of functional numerical testing as well.

**PLEASE NOTE!** If a script creates matplotlib graph windows, these windows need to be closed manually in order for test execution to continue.

