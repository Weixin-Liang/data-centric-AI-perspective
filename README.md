# Data-centric-AI-perspective

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

This repo provides the data and code for reproducing the the image classification experiments.

## Data
- Please download the prepared dataset `data-centric-AI-perspective.zip` from [this google drive url](https://drive.google.com/file/d/1UIEYbtNKp9sFOisGK1NPc49CLEEMZpVQ/view?usp=sharing).

- Extract the files. The resulting file structure will look like:

```plain
.
├── README.md
├── data/
    ├── train/              
        ├── cat/
            ├── <ID>.jpg
            ├── ...
        ├── dog/
            ├── <ID>.jpg
            ├── ...
    ├── val/
        ├── cat/
            ├── <ID>.jpg
            ├── ...
        ├── dog/
            ├── <ID>.jpg
            ├── ...
    ├── shapley_val/
        ├── cat/
            ├── <ID>.jpg
            ├── ...
        ├── dog/
            ├── <ID>.jpg
            ├── ...
├── code/
    ├── outputs/ 
    ├── step_1_baseline_finetune.ipynb
    ├── step_2a_extract_features.ipynb
    ├── step_2b_compute_data_shapley.py
    ├── step_3_using_cleaned_data.ipynb              
```




## Code
- Execute the scripts/notebooks in the following order, as indicated by their file names:
    - `step_1_baseline_finetune.ipynb`: performs standard fine-tuning on the given dataset. The notebook is adopted from the official PyTorch tutorial on [fine-tuning torchvision models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html). 
    - `step_2a_extract_features.ipynb`: extracts image features. The features are used when applying the data Shapley method to the noisy training set.
    - `step_2b_compute_data_shapley.py`: estimates the Shapley value of each training point. 
    - `step_3_using_cleaned_data.ipynb`: drops the training data points with negative Shapley value, and trains the classifier on the remaining training data points. 
    

## Dependencies
* Python 3.6.13 (e.g. `conda create -n venv python=3.6.13`)
* PyTorch Version:  1.4.0
* Torchvision Version:  0.5.0