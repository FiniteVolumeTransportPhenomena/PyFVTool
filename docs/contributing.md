# Contributing

Contributions are welcome — including documentation improvements!

## Setting up a development environment

### Create specific Conda environment

```bash
conda create --name pyfvtool_dev python=3.12 mkl numpy scipy matplotlib spyder jupyterlab pytest tqdm
conda activate pyfvtool_dev
conda install sphinx furo myst-parser nbsphinx pandoc
pip install sphinx-copybutton sphinx-autobuild
```

### Create a local git repo with an editable install 

```bash
conda activate pyfvtool_dev
git clone https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
cd PyFVTool
pip install --editable .
```

(Do not forget the trailing dot!)

## Building the docs locally

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
# Then open _build/html/index.html in a browser
```

Or use `sphinx-autobuild` for live reload during writing:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

## Writing documentation

- Pages are written in **Markdown** using [MyST](https://myst-parser.readthedocs.io/).
- Math: use `$...$` for inline and `$$...$$` for display equations.
- Code blocks: standard fenced code blocks with language tags (` ```python `).
- To add a new page, create a `.md` file and add it to the relevant `toctree`
  in `index.md` or the appropriate section index.

## Docstring style

PyFVTool uses **NumPy-style docstrings**. Example:

```python
def diffusionTerm(D_face):
    """
    Discretize the diffusion term ∇·(D ∇φ).

    Parameters
    ----------
    D_face : FaceVariable
        Diffusion coefficient defined at cell faces.

    Returns
    -------
    matrix_term : EquationTerm
        Sparse matrix contribution to be passed to `solvePDE`.
    """
```
