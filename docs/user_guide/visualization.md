# Visualization

PyFVTool provides simple built-in plotting via `matplotlib`.

## Plotting cell values

```python
pf.visualizeCells(c)
```

Works for 1D, 2D, and 3D cell variables. In 1D it produces a line plot;
in 2D a colour map; in 3D a slice or surface plot.

## Customizing plots

`visualizeCells` accepts optional `matplotlib` keyword arguments.
For more control, access the underlying numpy array and plot it directly:

```python
import matplotlib.pyplot as plt

plt.plot(mesh.cellcenters.x, c.value)
plt.xlabel('x [m]')
plt.ylabel('Concentration')
plt.show()
```

For 2D fields:

```python
plt.contourf(mesh.cellcenters.x, mesh.cellcenters.y, c.value.T)
plt.colorbar()
plt.show()
```
