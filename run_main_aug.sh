code_location="/home/u42/Windows/Data/datasets/SensorEffectAugmentation"
input="/home/u42/Documents/mmdetection/data/synth_rocks/train/"
output="/home/u42/Documents/mmdetection/data/synth_rocks/train_augmented/"

nvidia-docker run --rm -it \
  -v $input:/root/G_data \
  -v $output:/root/ResultsDir \
  -v $code_location/main_aug.py:/root/main.py \
  -v $code_location/model_aug.py:/root/model.py \
  -v $code_location/geometric_transformation_module.py:/root/geometric_transformation_module.py \
  -v $code_location/augmentfunctions_tf.py:/root/augmentfunctions_tf.py \
  -v $code_location/pix2pix_labtoRGBconv.py:/root/pix2pix_labtoRGBconv.py \
  -v $code_location/utils.py:/root/utils.py \
  tf-sensor-augment \
  python /root/main.py \
  -n 2 \
  --input /root/G_data \
  --output /root/ResultsDir \
  --image_height 480 \
  --image_width 720 \
  --exposure True \
  --noise True \
  --colour_shift True \

  2>&1 | tee -a sensortransfer-testing-logs.txt

