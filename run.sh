#!/bin/bash
#SBATCH -A pi_zy
#SBATCH -p gpu2Q
#SBATCH -q gpuq
#SBATCH --gres=gpu:2
python work_2.py
