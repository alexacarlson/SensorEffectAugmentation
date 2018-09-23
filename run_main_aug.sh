nvidia-docker run --rm -it \
  -v /mnt/ngv/askc-home/SensorTransfer_datasets/GTA5-clean/leftImg8bit/train-all-data:/root/G_data \
  -v `pwd`/GTA5_STdomrand:/root/ResultsDir \
  -v `pwd`/main_aug.py:/root/main.py \
  -v `pwd`/model_aug.py:/root/model.py \
  -v `pwd`/geometric_transformation_module.py:/root/geometric_transformation_module.py \
  -v `pwd`/augmentfunctions_tf.py:/root/augmentfunctions_tf.py \
  -v `pwd`/pix2pix_labtoRGBconv.py:/root/pix2pix_labtoRGBconv.py \
  -v `pwd`/utils.py:/root/utils.py \
  tf-sensor-augment \
  python /root/main.py \
  --Img_dataset /root/G_data \
  --Img_height 512 \
  --Img_width 1024 \
  --input_fname_pattern .png \
  --results_dir /root/ResultsDir \
  --epoch 1 \
  --save_aug_params_flag False \
  --blur_flag True \
  --chromab_flag True \
  --exposure_flag True \
  --noise_flag True \
  --color_flag True \

  2>&1 | tee -a sensortransfer-testing-logs.txt

