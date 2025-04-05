# AI tasks

This repository contains two tasks completed as part of the PSi course:

## Task 1 – Steam Games Data Analysis
File: `zad1.py`

Description:
- Loads data from a CSV file containing information about Steam games.
- Analyzes game availability on Windows/Mac/Linux platforms.
- Calculates the percentage of positive reviews.
- Uses K-means clustering based on price and review scores.
- Recommends similar games based on genre and price using KNN.

### Required libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Task 2 – Bear Image Classification
File: `zad2.py`

Description:
- Convolutional Neural Network (CNN) model for image classification (5 classes).
- Trains the model using image datasets in `data/train` and `data/val`.
- Evaluates model accuracy using test data in `data/test`.
- Saves and loads the PyTorch model from disk.

### Required libraries:
- PyTorch
- torchvision
- numpy
- matplotlib

## How to Run

### Task 1:
Make sure the path to `games.csv` in `zad1.py` is correct.

```bash
python zad1.py
