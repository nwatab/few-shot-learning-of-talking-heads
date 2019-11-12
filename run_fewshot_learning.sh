#!/bin/sh

#$ -l rt_G.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5
module load cuda/10.0/10.0.130.1
module load cudnn/7.4/7.4.2
module load nccl/2.4/2.4.7-1
source ~/dl/bin/activate

python fewshot_learning.py

