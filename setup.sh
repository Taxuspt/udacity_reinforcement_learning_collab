#!/bin/sh

conda create --name drlp3 python=3.8 -y
conda activate drlp3
conda install --file requirements.txt -y
pip install python/
