#!/bin/bash 

#SBATCH --output=lr_fixed_two_layer_100_hidden_100_embeddding.out
#SBATCH -t 0-3:30 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asn264@nyu.edu

module load pytorch/intel
module load torchvision

python main.py --emsize 100 --nhid 100 --nlayers 2 --epochs 15 --save lr_fixed_two_layer_100_hidden_100_embeddding.m