#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python student_net_learning/main.py --name Baseline1 --epochs 20 --batch_size 32 --datalist ../data/datalist/ --root ../data/ --cuda 1