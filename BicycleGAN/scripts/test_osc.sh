set -ex
# models
RESULTS_DIR='./results/osc'

# dataset
CLASS='osc'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=286 # scale images to this size
FINE_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
HOW_MANY=10 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

NAME='npy_dataset_osc_13_09_2018_14'
NZ=16
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test_osc.py \
  --dataroot /home/solomon/im2im/BicycleGAN/datasets/npy_dataset \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir /home/solomon/im2im/checkpoints/npy_dataset \
  --output_nc 4 \
  --nz ${NZ} \
  --dataset_mode npz \
  --name ${NAME} \
  --model ${CLASS}
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} \
  --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --conditional_D
