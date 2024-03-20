# Personal Installation Guide

#### This recipe works for ubuntu 22.04 ✅
 
This guide outlines the necessary steps to set up a development environment tailored for scientific computing and robotics development on Linux. Follow these instructions to ensure you have all the necessary tools and libraries installed.

## Initial System Update and Package Installation

First, update your system's package list and upgrade any existing packages to their latest versions. Then, install essential tools, libraries, and Python packages needed for development.

Update system package list and upgrade existing packages
```
sudo apt update && sudo apt upgrade -y
```

Install cmake and essential C/C++ development tools
```
sudo apt install -y cmake build-essential cmake-curses-gui libeigen3-dev libboost-all-dev libgl1
```

Install Python development tools and essential scientific libraries
```
sudo apt install -y cython3 python3-pip
```

## Python Package Installation via pip

Next, install additional Python packages required for scientific computing and robotics development. Ensure you are using `pip3` for Python 3.x compatibility.

Install PyTorch
```
pip3 install torch
```

Install additional scientific and robotics development packages
```
pip3 install chumpy vctoolkit open3d pybullet qpsolvers cvxopt
```

Also install other packages
```
pip3 install numpy==1.23.1 scipy pyyaml qpsolvers[quadprog]
```

## Installing rbdl from Source

`rbdl` is a library for rigid body dynamics. Follow these steps to install it from source.

### Clone the rbdl Repository

First, clone the `rbdl` repository using `git`. Use the `--recursive` option to ensure all submodules are also cloned.

```bash
git clone --recursive https://github.com/rbdl/rbdl
```

### Build rbdl

Create a build directory for `rbdl`, configure the project with `cmake`, and then compile it.

Create and enter the build directory
```bash
mkdir rbdl-build && cd rbdl-build/
```
Configure the project
```
cmake -D CMAKE_BUILD_TYPE=Release ../rbdl
```

Open ccmake to select build options
```
ccmake ../rbdl
```

In the ccmake UI, set:
```
RBDL_BUILD_ADDON_URDFREADER to ON
RBDL_BUILD_PYTHON_WRAPPER to ON
```
Then, press 'c' to configure, followed by 'g' to generate and exit

Then compile the project
```
make
```

### Finalizing the Installation

Add the `rbdl` Python library to your `PYTHONPATH` to make it accessible to your Python scripts.

```bash
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/rbdl-build/python
```

## Upload Project
```
sudo scp -r -i "my_new_key.pem" /Users/xxxx/Documents/GitHub/PIP ubuntu@ec2-54-219-201-246.us-west-1.compute.amazonaws.com:/home/ubuntu/
```

## Usage

After completing the installation steps, you can run your Python scripts that depend on the installed libraries and tools.

```bash
python3 evaluation.py
```