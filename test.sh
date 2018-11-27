#!/bin/bash


rm -rf save
mkdir save
python generate.py -c sn_full_17 -e sn_full_17 -t sn
python evaluate.py -p save -g ~/datasets/eecs442challenge/train/normal/
