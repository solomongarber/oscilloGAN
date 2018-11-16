set -ex
CLASS='supervised_onehot'  #'npy_dataset' # maps' # edges2shoes, facades, day2night, edges2shoes, edges2handbags, maps
#echo ${CLASS}
MODEL='osc'
#CLASS=${1}
GPU_ID=2
DISPLAY_ID=$((GPU_ID*10+1))
PORT=8097
NZ=16
INPUT_NZ=32
echo ${CLASS}
CHECKPOINTS_DIR=../checkpoints/${CLASS}/
DATE=`date '+%d_%m_%Y_%H'`
NAME=${CLASS}_${MODEL}_${DATE}


# dataset
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE=286
FINE_SIZE=256
INPUT_NC=3

# dataset parameters
case ${CLASS} in
'facades')
  NITER=200
  NITER_DECAY=200
  SAVE_EPOCH=25
  DIRECTION='BtoA'
  ;;
'edges2shoes')
  NITER=30
  NITER_DECAY=30
  LOAD_SIZE=256
  SAVE_EPOCH=5
  INPUT_NC=1
  NO_FLIP='--no_flip'
  ;;
'edges2handbags')
  NITER=15
  NITER_DECAY=15
  LOAD_SIZE=256
  SAVE_EPOCH=5
  INPUT_NC=1
  ;;
'maps')
  NITER=200
  NITER_DECAY=200
  LOAD_SIZE=600
  SAVE_EPOCH=25
  DIRECTION='BtoA'
  ;;
'npy_dataset')
  NITER=200
  NITER_DECAY=200
  SAVE_EPOCH=20
  ;;
'supervised_onehot')
  NITER=200
  NITER_DECAY=200
  SAVE_EPOCH=50
  ;;  
'day2night')
  NITER=50
  NITER_DECAY=50
  SAVE_EPOCH=10
  ;;
*)
  echo 'WRONG category'${CLASS}
  ;;
esac



# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot /home/solomon/im2im/BicycleGAN/datasets/${CLASS} \
  --dataset_mode freq\
  --output_nc 4\
  --name ${NAME} \
  --model ${MODEL} \
  --display_port ${PORT} \
  --which_direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nz ${NZ} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --conditional_D \
  --niter_decay ${NITER_DECAY} \
  --input_nz ${INPUT_NZ} \
  --lambda_G2 1.0 \
  --use_dropout \
  --save_epoch_freq ${SAVE_EPOCH}
  --use_mse
