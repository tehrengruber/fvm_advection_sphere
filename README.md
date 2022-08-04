# Installing PyVista

Python 3.10 (required by GT4Py functional frontend) is not supported by the VTK
version on PyPi right now (April 2022). To avoid this build VTK from source.

```bash
mkdir vtk_build
cd vtk_build
git clone https://gitlab.kitware.com/vtk/vtk.git vtk_source
git checkout a52377af  # check https://gitlab.kitware.com/vtk/vtk/-/commits/master for a working commit
cmake -GNinja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON vtk_source
ninja
pip install .
```