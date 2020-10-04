#!/usr/bin/env bash

pip uninstall pillow
conda uninstall --force jpeg libtiff -y
conda install -c conda-forge libjpeg-turbo
CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall --no-binary :all: --compile pillow-simd
