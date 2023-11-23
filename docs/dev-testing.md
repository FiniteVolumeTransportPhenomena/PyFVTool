# PyFVTool development: testing

Simple automated testing of the Python code can be done using `pytest` for which a configuration has been set up. The configuration has been kept minimalist. It enables to conveniently test if the PyFVTool still works as expected before commiting changes to the code repository (although 100 percent code testing coverage is not guarantueed at this stage). 

All available tests can be run by simpling invoking
``   
pytest
``  
from the command line, when in the `pyfvtool` project root directory.

Running tests requires to have installed in your Python environment:
- `pytest`
- `pytest_notebook` (**mind the underscore**, *do not use a dash*!)

The latter can be installed, *e.g.*, with `pip` or `mamba install pytest pytest_notebook` (**mind the underscore**, *do not use a dash*!). If you do not have `pytest_notebook` and Jupyter Notebook available, it should be possible to run `pytest` nevertheless by removing the lines starting with `nb_` from the `pytest.ini` file

The present pytest configuration scans all directories for files named `test_*.py` or `*_test.py`, or `*.ipynb` (Notebooks). These will be considered "tests". Several tests compare the finite-volume result to the known analytic result for textbook cases, and thus provide a rudimentary form of functional numerical testing as well.

**PLEASE NOTE!** If a script creates matplotlib graph windows, these windows need to be closed manually in order for test execution to continue.

