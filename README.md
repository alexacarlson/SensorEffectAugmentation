##
## SENSOR EFFECT AUGMENTATION PIPELINE 

This repository contains the tensorflow implementation of the Sensor Effect Augmentation Pipeline described in *Modeling Camera Effects to Improve Visual Learning from Synthetic Data* (https://arxiv.org/abs/1803.07721).

### Setting up the Sensor Effect Pipeline
This pipeline uses docker/containers, so it can be run on any computer. 
For a good tutorial on docker see https://docs.docker.com/get-started/.
The docker image used by this model needs to be built before being used, which is done by running the following 
command in the Sensor_Augmentation_Pipeline folder:

```$docker build -t tf-sensor-augment augment-docker-image/```

where ```tf-sensor-augment``` is the tag for the docker image and is what is used to start a container.
The docker image is specified in ```augment-docker-image/Dockerfile```.


### Running the Pipeline/Augmenting Images

To run the pipeline, use the command

```$. run_main_aug.sh```

in the command line within the SensorEffectAugmentation folder.
You can customize the type of augmentations, the dataset to augmented, etc by modifying the ```run_main_aug.sh```, which is described in more detail below.

The Sensor Effect Image Augmentation Pipeline is comprised of the following files:

* ```run_main_aug.sh```

   This is a bash file that initializes a docker container and runs the image augmentation pipeline.

   The variable ```input``` is the path to the directory that holds the images you would like to augment
   
   The variable ```output``` is the path to the directory where the augmented images will be saved. The default location is set to ```results``` in the SensorEffectAugmentation directory.
   
   The variable ```code_location``` is the path to the directory where SensorEffectAugmentation is located

   The other volume mappings are setting up the file system in the docker container. 
   
   The command ```python /root/main.py``` (line 15) runs the image augmentation pipeline. 
	```bash
	usage: main.py [-h] [-n [N]] [-b [BATCH_SIZE]] [-c [CHANNELS]] [-i INPUT]
		       [-o [OUTPUT]] [--pattern [PATTERN]]
		       [--image_height [IMAGE_HEIGHT]] [--image_width [IMAGE_WIDTH]]
		       [--chromatic_aberration [CHROMATIC_ABERRATION]] [--blur [BLUR]]
		       [--exposure [EXPOSURE]] [--noise [NOISE]]
		       [--colour_shift [COLOUR_SHIFT]] [--save_params [SAVE_PARAMS]]

	Augment a dataset

	optional arguments:
	  -h, --help            show this help message and exit
	  -n [N]                sets the number of augmentations to perform on the
				dataset i.e., setting n to 2 means the dataset will be
				augmented twice
	  -b [BATCH_SIZE], --batch_size [BATCH_SIZE]
				size of batches; must be a multiple of n and >1
	  -c [CHANNELS], --channels [CHANNELS]
				dimension of image color channel (note that any
				channel >3 will be discarded
	  -i INPUT, --input INPUT
				path to the dataset to augment
	  -o [OUTPUT], --output [OUTPUT]
				path where the augmented dataset will be saved
	  --pattern [PATTERN]   glob pattern of filename of input images
	  --image_height [IMAGE_HEIGHT]
				size of the output images to produce (note that all
				images will be resized to the specified image_height x
				image_width)
	  --image_width [IMAGE_WIDTH]
				size of the output images to produce. If None, same
				value as output_height
	  --chromatic_aberration [CHROMATIC_ABERRATION]
				perform chromatic aberration augmentation
	  --blur [BLUR]         perform blur augmentation
	  --exposure [EXPOSURE]
				perform exposure augmentation
	  --noise [NOISE]       perform noise augmentation
	  --colour_shift [COLOUR_SHIFT]
				perform colour shift augmentation
	  --save_params [SAVE_PARAMS]
				save augmentation parameters for each image
	```
* ```main_aug.py```

	This is a master function that handles input flags and initializing the augmentation. It is called by ```run_main_aug.sh```.

* ```model_aug.py```

	This is a python module that defines the image augmentation class and builds the tensorflow graph. It is called by ```main_aug.py```. To alter the sensor effect parameter ranges, you will need to change the values in the ```generate_augmentation``` function, lines 123-200 in this file.

* ```augmentationfunctions_tf.py```

	This is a python module that contains all of the sensor augmentation functions for blur, chromatic aberration, exposure changes, sensor noise, and color shifts.
	Please see our arxiv paper (https://arxiv.org/abs/1803.07721) for more information on these functions.

* ```geometric_transformation_module.py```

	This is a python module that implements affine warping; is used for augmenting images with chromatic aberration via warping the R and B color channels relative to the G channel.
	It is called by augmentationfunctions_tf.py

* ```pix2pix_lab2RGBconv.py```

	This is a python module that implements color conversion between RGB and LAB color spaces. It is called by augmentationfunctions_tf.py

Examples of GTA images augmented with the above sensor effects are located in ```Sensor_Augmentation_Pipeline/augFIGURES``` folder.




