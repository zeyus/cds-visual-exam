# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset.

You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.

## Tips

- You should structure your project by having scripts saved in a folder called ```src```, and have a folder called ```out``` where you save the classification reports.
- Consider using some of the things we've seen in class, such as virtual environments and setup scripts.

## Purpose

- To ensure that you can use ```scikit-learn``` to build simple benchmark classifiers on image classification data
- To demonstrate that you can build reproducible pipelines for machine learning projects
- To make sure that you can structure repos appropriately
