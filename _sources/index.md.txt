# PyFVTool

**PyFVTool** is a Python toolbox for solving convection-diffusion-reaction equations
using the [finite volume method](https://en.wikipedia.org/wiki/Finite_volume_method) (FVM).

It discretizes and numerically solves the conservative form of transient PDEs of the
general form:

$$
	\alpha\frac{\partial\phi}{\partial t}
	+ \nabla \cdot \left(\mathbf{u}\phi\right)
	+ \nabla \cdot (-\mathcal{D}\nabla\phi)
	+ \beta\phi
	+ \gamma = 0
$$

with general boundary conditions given as 

$$
a\nabla\phi \cdot \mathbf{e} + b\phi = c
$$

PyFVTool is oriented toward **heat and mass transport phenomena** on structured meshes,
for cases where the flow velocity field is already known or absent. It is pure scientific
Python. Only `numpy`, `scipy`, and `matplotlib` are required.

---

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/meshes
user_guide/coordinate_labels
user_guide/cell_variables
user_guide/boundary_conditions
user_guide/discretization
user_guide/solving
user_guide/visualization
user_guide/functions_and_classes
```

```{toctree}
:maxdepth: 1
:caption: Examples

examples/index
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
changelog
origin_story
```

