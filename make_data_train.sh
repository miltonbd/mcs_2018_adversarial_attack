
#!/bin/sh
python prepare_data.py --root ../data/student_model_imgs/ --cuda 1--datalist_path ../data/datalist/ --datalist_type train --gpu_id 1 --size 50000;