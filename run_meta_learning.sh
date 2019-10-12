#!/bin/sh

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd


source /etc/profile.d/modules.sh
module load python/3.6/3.6.5
module load cuda/10.0/10.0.130.1
module load cudnn/7.4/7.4.2
module load nccl/2.4/2.4.7-1
source ~/dl/bin/activate

python run_meta_learning.py
#NUM_NODES=${NHOSTS}
#NUM_GPUS_PER_NODE=4
#NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
#NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})

#MPIOPTS="-n ${NUM_PROCS}"

#APP = "python /home/aca10485tl/work/few-shot-learning-of-talking-heads/meta_learning.py"

#horovodrun ${MPIOPTS} ${APP}
