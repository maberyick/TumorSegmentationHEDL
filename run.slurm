#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=0-20:00:00
#SBATCH --mail-user=cfk29@case.edu
#SBATCH -p gpu -C gpuk40 --gres=gpu:1
#SBATCH --job-name="tumor mask"

#============parameters===========#
#PathWSI='/mnt/rstor/CSE_BME_AXM788/data/Kaisar_OP/Ventana_KA_Slides/April2020/'
#PathWSI='/mnt/rstor/CSE_BME_AXM788/data/UH_NSCLC_Pathology/IO/NSCLC Ventana Images MAR2020/'
PathWSI='/mnt/rstor/CSE_BME_AXM788/data/BMS_LungIO_Path/CA209-057/'
FileExtWSI='.svs'

#TumorMaskPathWSI='/mnt/rstor/CSE_BME_AXM788/data/Vanderbilt_Oropharyngeal_WSI/Jims_scans/Philips/tumorMask/'
#TumorMaskPathWSI='/scratch/users/cfk29/Kaisar_OP/tumorMask/'
#TumorMaskPathWSI='/scratch/users/cfk29/UH_NSCLC/'
TumorMaskPathWSI='/scratch/users/crb138/bms/tumor_dlmask/'

PathPatch='/scratch/users/cfk29/Vanderbilt_p16/patches2048_40x/'
NucleiDlPatch='/scratch/users/cfk29/Houston_VA_MN_results_all/'
CellSegPatch='/scratch/users/cfk29/Houston_VA_MN_seg_all/'

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

pip2 install --upgrade opencv-python --user
pip2 install --upgrade scikit-image==0.13.1 --user
pip2 install --upgrade numpy==1.15.0 --user
pip2 install openslide-python --user

time python2 ./src/arg_256_wsi_clc_cheng_v2.py --num1 $StartIndex --num2 $EndIndex --deploy $TumorDeployPath --model $TumorModelPath --mean_proto_path $TumorMeanProtoPath --TUMOR_WSI_PATH "${PathWSI}" --HEAT_MAP_SAVE_PATH $TumorMaskPathWSI --WSI_EXT $FileExtWSI

exit 0

#=========patch extraction======#
module load python/3.6.6

python3.6 $HOME/pipeline/PatchExtraction/src/extractPatches.py $PatchSize $ImageLevel $PathWSI $WSIFileExt $PathPatch $StartIndex $EndIndex $TumorMaskPathWSI

#=========nuclei segmentation======#
#module load python2
#pip install --user http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
#pip install --user torchvision
module load python2
module load matlab

#python2 $HOME/pipeline/NucleiSegmentation/src/dl_nuclei_seg.py --trained_model $NucleiModelPath --image_size $patchSize --dataroot $patchesDataRoot --results_dir $nucleiDlOutput --start_index $startInd --end_index $endInd
$HOME/pipeline/NucleiSegmentation/src/deal_with_nuclei_clump_HPC_v3 $startInd $endInd $dlOutput $cellSegOutput
