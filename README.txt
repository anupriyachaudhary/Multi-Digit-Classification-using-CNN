REQUIREMENTS:
Download following files from http://ufldl.stanford.edu/housenumbers/ :
1. Full Numbers: train.tar.gz, test.tar.gz , extra.tar.gz and extra these files in the ‘data/full’ folder
2. Cropped Digits: train_32x32.mat, test_32x32.mat , extra_32x32.mat and place all the files in the ‘data/cropped’ folder.

FILES & FOLDERS:
(1) extract_neg_from_full.py: extracts negative images (32x32) from FULL dataset for training with cropped images and creates train_neg.h5. & test_neg.h5 in ‘data/cropped’ folder
(2) preprocess_cropped.py: prepares train, validation and test data from CROPPED dataset (together with negative images created using extract_neg_from_full.py) and creates train.h5, valid.h5 & test.h5 in ‘data/cropped’ folder
(3) train_model.py: contains code to train on CROPPED images
(4) pre_proc_full.py: prepares train, validation and test data from FULL dataset (together with negative images) and creates train.h5 & test.h5 in ‘data/cropped’ folder
(5) train_model_multi.py: contains code to train on FULL images
(6) model.h5: model final chosen to do detection
(7) images: this folder contains images to carry out detection
(8) run.py: writes five images to a graded_images folder in the current directory


PART I: MODEL CREATED ON CROPPED DATASET

Test Instructions:
(1) Run following script to extract negative images (32x32) from FULL dataset
	$ python extract_neg_from_full.py

(2)To create train, validation and test data run the following command:
	$ python preprocess_cropped.py

(3) To train the training data run train_model.py
	$ python train_model.py

(4) To run detection on chosen images run following file
	$ python run.py


PART II: MODEL CREATED ON FULL DATASET

Test Instructions:
(1) To create train, validation and test data run the following command:
	$ python pre_proc_full.py

(3) To train the training data run following
	$ python train_model_multi.py

(4) To run detection on chosen images run following file
	$ python run.py
