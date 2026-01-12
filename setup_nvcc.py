#! /usr/bin/env python
#
# Setup nvcc compiler for building AnuGA with CUDA support

descr = """A set of python modules for modelling the effect of tsunamis and flooding"""

import sys
import os

os.environ["PROJECT_ROOT"] = os.getcwd()
os.environ["CC"] = "nvc -O3 -acc=gpu -Minfo=accel -noswitcherror -lm -I$CUDA_HOME/include/ --device-debug --generate-line-info --arch=sm_11 "
os.environ["CXX"] = "nvc++ -O3 -acc=gpu -Minfo=accel -noswitcherror -lm -I$CUDA_HOME/include/ --device-debug --generate-line-info -lineinfo --arch=sm_11 -std=c++17"
os.environ["FC"] = "nvfortran -O3 -acc=gpu -Minfo=accel -noswitcherror -lm -I$CUDA_HOME/include/ --device-debug --generate-line-info "

