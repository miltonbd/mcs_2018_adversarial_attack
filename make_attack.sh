#!/bin/sh
python3 attacker.py --root ../data/imgs/ --save_root ./baseline1/ --datalist ../data/pairs_list.csv --model_name ResNet18 --checkpoint_path checkpoint/Baseline1/best_model_chkpt.t7 --cuda