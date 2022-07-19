#!/bin/bash

rm -rf venv_pwn/
virtualenv --system-site-packages -p python3.8 ./venv_pwn
source ./venv_pwn/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# make dir for plots and models
mkdir ./res/plots
mkdir ./res/models
mkdir ./res/experiments
