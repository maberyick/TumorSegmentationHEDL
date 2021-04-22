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
PathWSI='/mnt/rstor/CSE_BME_AXM788/data/BMS_LungIO_Path/CA209-057/'
FileExtWSI='.svs'
TumorMaskPathWSI='/scratch/users/crb138/bms/tumor_jpgmask/'

StartIndex=0
EndIndex=-1
PatchSize=2048
ImageLevel=0

TumorDeployPath='./model/deploy_vgg16.prototxt'
TumorModelPath='./model/vgg-ocal-5-1_iter_35000.caffemodel'
TumorMeanProtoPath='./model/train_mean_3.binaryproto'
NucleiModelPath='../checkpoints/multi_nucleation/latest_net_G.pth'

#======tumor segmentation=========#
module load gcc/6.3.0
module load openslide
module load caffe
module load cuda/8.0
module load python2

time python2 ./src/arg_256_wsi_clc_cheng_v2.py --num1 $StartIndex --num2 $EndIndex --deploy $TumorDeployPath --model $TumorModelPath --mean_proto_path $TumorMeanProtoPath --TUMOR_WSI_PATH "${PathWSI}" --HEAT_MAP_SAVE_PATH $TumorMaskPathWSI --WSI_EXT $FileExtWSI

exit 0
