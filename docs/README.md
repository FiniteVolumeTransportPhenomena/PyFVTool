# PyFVTool documentation

In the present, pre-release state of PyFVTool, documentation is scarce. A documentation system is being set-up using Sphinx. Docstrings are being added to the code, also enabling the `help` function in interactive Python sessions. 

The `examples-notebooks` folder of the PyFVTool repository contains examples of how to set up finite-volume solvers using PyFVTool for certain textbook cases. 

Further documents will be collected here. Initially as simple MarkDown text files, which can later be integrated into automated documentation management.

## Sphinx

See `CONTRIBUTING.md` for installation of the development environment, which also includes Sphinx and its plug-ins.

### Sphinx set-up

Sphinx has been set up as follows. The Sphinx set-up procedure is recorded here for future reference, but needs not be repeated, as the `docs` folder has already been configured.

```
sphinx-quickstart
```

`docs/conf.py` has been modified.


### Building the documentation

From within the `docs` folder

```
make clean
make html
```



