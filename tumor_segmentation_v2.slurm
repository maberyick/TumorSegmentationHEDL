#!/bin/bash
#SBATCH --job-name="tumor_mask"
#SBATCH -o log/tumor_seg_%A_%a.out
#SBATCH -e log/tumor_seg_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=0-20:00:00
#SBATCH --mail-user=crb138@case.edu
#SBATCH -p gpu -C gpuk40 --gres=gpu:1
#============parameters===========#
cd /home/crb138/preprocessing/tumor_segmentation
PathWSI='/scratch/users/crb138/primary_mets/wsi/'
FileExtWSI='.tiff'
TumorMaskPathWSI='/scratch/users/crb138/primary_mets/tumor_dlmask/'

StartIndex=0
EndIndex=-1
PatchSize=2048
ImageLevel=0

TumorDeployPath='/home/crb138/preprocessing/tumor_segmentation/model/deploy_vgg16.prototxt'
TumorModelPath='/home/crb138/preprocessing/tumor_segmentation/model/vgg-ocal-5-1_iter_35000.caffemodel'
TumorMeanProtoPath='/home/crb138/preprocessing/tumor_segmentation/model/train_mean_3.binaryproto'

#======tumor segmentation=========#
module load gcc/6.3.0
module load openslide
module load caffe
module load cuda/8.0
module load python2

time python ./src/arg_256_wsi_clc_cheng_v2.py --num1 $StartIndex --num2 $EndIndex --deploy $TumorDeployPath --model $TumorModelPath --mean_proto_path $TumorMeanProtoPath --TUMOR_WSI_PATH "${PathWSI}" --HEAT_MAP_SAVE_PATH $TumorMaskPathWSI --WSI_EXT $FileExtWSI
exit 0
