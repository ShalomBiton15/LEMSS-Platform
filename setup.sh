#!/bin/bash
# One time setup of the work environment

# 1. Miniconda in ~/PROGS
mkdir -p ~/PROGS && cd ~/PROGS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/PROGS/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
source ~/PROGS/miniconda3/bin/activate
conda init
conda config --set auto_activate_base false

# 2. Project environment 
# conda env create -f environment.yml
