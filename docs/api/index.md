# API reference

This section documents the public API of PyFVTool. The text is generated automatically from the source code docstrings.

*This section is in a preliminary state. For now, the code itself is the best documentation. Nonetheless, this section is a decent entry point.*

## To do:

- update the `automodule` directives (in `docs/api/index.md`) to match the
the actual filenames in the `src/pyfvtool/`.

- `docs/api/index.md` is just a plain Markdown file, it can be edited freely. Add the missing modules and restructure however makes sense.




## `pyfvtool` (top-level)

There's nothing here, as there are no definitions in `__init__.py`, only imports. **TODO** We may remove this section.

```{eval-rst}
.. automodule:: pyfvtool
   :members:
   :undoc-members: False
```




## Meshes

```{eval-rst}
.. automodule:: pyfvtool.mesh
   :members:
   :undoc-members: False
```

## Cell and face variables

### Cell variables

```{eval-rst}
.. automodule:: pyfvtool.cell
   :members:
   :undoc-members: False
```

### Face variables

```{eval-rst}
.. automodule:: pyfvtool.face
   :members:
   :undoc-members: False
```


## Boundary conditions

```{eval-rst}
.. automodule:: pyfvtool.boundary
   :members:
   :undoc-members: False
```

## Discretization terms

### Diffusion terms

```{eval-rst}
.. automodule:: pyfvtool.diffusion
   :members:
   :undoc-members: False
```

### Advection terms

```{eval-rst}
.. automodule:: pyfvtool.advection
   :members:
   :undoc-members: False
```

### Source terms

```{eval-rst}
.. automodule:: pyfvtool.source
   :members:
   :undoc-members: False
```





## Solver

```{eval-rst}
.. automodule:: pyfvtool.pdesolver
   :members:
   :undoc-members: False
```

## Averaging utilities

```{eval-rst}
.. automodule:: pyfvtool.averaging
   :members:
   :undoc-members: False
```

## Visualization

```{eval-rst}
.. automodule:: pyfvtool.visualization
   :members:
   :undoc-members: False
```
