# VGG16 Image Classification with the Indo Fashion Dataset

This repository contains code that fine tunes a VGG16 model on the Indo Fashion dataset. The training history plot, confusion matrix and classification report are saved in the `out/` folder, along with a csv file containing the training history, and a txt file containing the model summary and classification report.

The data were preprocessed using the following steps:

- Proportionally rescaled the images to `IMAGE_SIZE`x`IMAGE_SIZE` pixels, or `IMAGE_WIDTH`x`IMAGE_HEIGHT` pixels if specified.
- Images were zero-padded to match the input size of the model, with the image centered.
- Normalized the pixel values to be between 0 and 1.
- For training data, random horizontal flips were and random rotations of -20 to 20 percent were applied.

For the results below, the images were resized to 100x200 pixels, and the batch size was 256.

## Setup

### Windows GPU (optional)

If you have an NVIDIA GPU you can do the following before installing the prerequisites:

- Install [Anaconda](https://www.anaconda.com/products/individual)
- Create a new environment using `conda create -n vgg16 python=3.9`
- Activate the environment using `conda activate vgg16`
- Install cudatoolkit and cudnn with `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

### Prerequisites

- Install the required packages using `pip install -r requirements.txt`
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset), unzip and save it to `data/`
- - Get your kaggle API token from [here](https://www.kaggle.com/settings) and save it to `~/.kaggle/kaggle.json`
- - Run `python src/cnn.py --download` to download the dataset

## Usage

- Run `python src/cnn.py` to train the model with the default parameters.

Most settings can be customized, such as input resizing, batch size, number of epochs, etc. Run `python src/cnn.py --help` to see all the available options.

```bash
❯ python .\src\cnn.py --help
usage: cnn.py [-h] [--version] [-m MODEL_SAVE_PATH] [--download] [-d DATASET_PATH] [-s IMAGE_SIZE] [-w IMAGE_WIDTH] [-t IMAGE_HEIGHT] [-b BATCH_SIZE] [-e EPOCHS] [-o OUT] [-n] [-c FROM_CHECKPOINT] [-r] [-p PARALLEL]

Text classification CLI

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -m MODEL_SAVE_PATH, --model-save-path MODEL_SAVE_PATH
                        Path to save the trained model(s) (default: models)
  --download            Download the dataset from kaggle (default: False)
  -d DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset (default: data)
  -s IMAGE_SIZE, --image-size IMAGE_SIZE
                        The image size (width and height) (default: 32)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs (default: 10)
  -o OUT, --out OUT     The output path for the plots and stats (default: out)
  -n, --no-train        Do not train the model (default: False)
  -c FROM_CHECKPOINT, --from-checkpoint FROM_CHECKPOINT
                        Use the checkpoint at the given path (default: None)
  -r, --resnet          Use ResNet50 as the base model. (default: False)
  -p PARALLEL, --parallel PARALLEL
                        Number of workers/threads for processing. (default: 4)
```

# Results

The final VGG16 model was trained in two steps, with a bach size of 256 and 150 epochs each.

The classification report, model history and confusion matrix are described below.

## First training step

### Classification report

```bash
precision    recall  f1-score   support

              blouse       0.93      0.89      0.91       500
         dhoti_pants       0.84      0.50      0.63       500
            dupattas       0.74      0.54      0.62       500
               gowns       0.71      0.45      0.55       500
           kurta_men       0.67      0.81      0.73       500
leggings_and_salwars       0.59      0.74      0.66       500
             lehenga       0.86      0.78      0.82       500
         mojaris_men       0.86      0.84      0.85       500
       mojaris_women       0.84      0.86      0.85       500
       nehru_jackets       0.90      0.81      0.85       500
            palazzos       0.88      0.63      0.74       500
          petticoats       0.64      0.86      0.73       500
               saree       0.60      0.91      0.72       500
           sherwanis       0.84      0.70      0.76       500
         women_kurta       0.53      0.74      0.62       500

            accuracy                           0.74      7500
           macro avg       0.76      0.74      0.74      7500
        weighted avg       0.76      0.74      0.74      7500
```

### Model history

![Model history](out/20230420_212902_vgg16_100x200_batch256_iter150.history.png)

### Confusion matrix

![Confusion matrix](out/20230420_212902_vgg16_100x200_batch256_iter150.cm.png)

## Second training step

### Classification report

```bash
precision    recall  f1-score   support

              blouse       0.90      0.94      0.92       500
         dhoti_pants       0.85      0.55      0.67       500
            dupattas       0.79      0.47      0.59       500
               gowns       0.69      0.41      0.52       500
           kurta_men       0.66      0.87      0.75       500
leggings_and_salwars       0.66      0.75      0.70       500
             lehenga       0.90      0.81      0.86       500
         mojaris_men       0.89      0.79      0.84       500
       mojaris_women       0.81      0.90      0.85       500
       nehru_jackets       0.90      0.80      0.85       500
            palazzos       0.89      0.68      0.77       500
          petticoats       0.75      0.83      0.79       500
               saree       0.71      0.91      0.80       500
           sherwanis       0.87      0.68      0.77       500
         women_kurta       0.45      0.88      0.60       500

            accuracy                           0.75      7500
           macro avg       0.78      0.75      0.75      7500
        weighted avg       0.78      0.75      0.75      7500
```

### Model history

![Model history](out/20230421_154647_vgg16_100x200_batch256_iter150.history.png)

### Confusion matrix

![Confusion matrix](out/20230421_154647_vgg16_100x200_batch256_iter150.cm.png)

# Conclusion

The model was able to achieve an accuracy of 75% on the test set, which is not bad, but it could have been better. 
Notably, something must have gone wrong with the training continuation, although the fine-tuned model was loaded
correctly, the model training did not seem to continue, but given more epochs it's possible that the accuracy cloud
have been improved.