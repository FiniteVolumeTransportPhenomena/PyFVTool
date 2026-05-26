# PyFVTool documentation

A documentation system using Sphinx has been set-up. Docstrings are being added to the code, also enabling the `help` function in interactive Python sessions. 

The [`/docs/source/notebook-examples/`](source/notebook-examples/) folder of the PyFVTool repository contains examples of how to set up finite-volume solvers using PyFVTool for a number of textbook cases. These Jupyter Notebooks are also integrated in the Sphinx documentation system.



### Building the documentation

From within the `docs` folder

```bash
sphinx-build -b html . _build/html
```

This will build everything, including the auto-API documentation, and the Jupyter notebooks (which will just be rendered, not explicitly re-run).

The documentation in HTML will be in `docs/_build/html/`. Just open `docs/_build/html/index.html` in a web browser.

It is also possible to use live-reload while writing:
```bash
# pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

## Documentation TO-DO

- Improve docstrings
- Tutorial documentation can nicely be written as Jupyter Notebooks.
  	- inspiration may be drawn from https://miepython.readthedocs.io -- [(source)](https://github.com/scottprahl/miepython)
  	- and https://sfepy.org -- [(source)](https://github.com/sfepy/sfepy)
- Do not worry about tweaking Sphinx. This will happen as we go. Keep it simple!
  	- see also: https://the-ultimate-sphinx-tutorial.readthedocs.io


## Sphinx included in the development Conda environment

See `CONTRIBUTING.md` for installation of the development environment, which also includes Sphinx and its plug-ins.






