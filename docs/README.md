# PyFVTool documentation

In the present, pre-release state of PyFVTool, documentation is scarce. A documentation system is being set-up using Sphinx. Docstrings are being added to the code, also enabling the `help` function in interactive Python sessions. 

The `/docs/source/notebook-examples/` folder of the PyFVTool repository contains examples of how to set up finite-volume solvers using PyFVTool for certain textbook cases. These Notebooks are also part of the Sphinx documentation.

Further documents will be collected here (in particular in `/docs/source/`). Markdown may be used.


## Documentation TO-DO

- Address all Sphinx `make clean; make html` warnings. Obtain a 'clean' output.
- Structure the TOC tree well (e.g. Examples in a subtree)
	- https://documentation.help/Sphinx/toctree.html
- Set up a way of publishing the built documentation (ReadTheDocs? github.io? PDF?)
- Tutorial documentation can nicely be written as Jupyter Notebooks.
- Do not worry about tweaking Sphinx. This will happen as we go. Keep it simple!


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

#### Regenerate RST source files for API documentation from PyFVTool docstrings

Can perhaps be automated? Or only needs to be run once, with further manual editing.

From within the `docs` folder

```
sphinx-apidoc ../src/pyfvtool -o ./source/_pyfvtool_autodoc
```



