# oneFEM

`oneFEM` is a Python package for finite element modeling and analysis. It includes various modules for performing simulations, performing mathematical operations, and visualizing the results.

## Installation

You can install `oneFEM` using `pip`:

```bash
pip install oneFEM

# umfpack setup #
pip install wheel
pip install meson meson-python
sudo apt install ninja-build
sudo apt install swig
sudo apt install libopenblas-dev liblapack-dev
sudo apt install pkg-config
sudo apt install libsuitesparse-dev
find /usr /opt -name umfpack.h
sudo apt install build-essential ninja-build swig python3-dev python3-numpy-dev
pip install numpy==1.26.4
pip install --force-reinstall --no-deps scipy scikit-umfpack