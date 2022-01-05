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
    ├── step_1_baseline_regular_train.ipynb
    ├── step_2a_extract_features.ipynb
    ├── step_2b_compute_data_shapley.py
    ├── step_3_using_cleaned_data.ipynb              
```




## Code
- Execute the scripts/notebooks step by step as indicated by their file names. 