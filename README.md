# Bachelor Thesis - The power of compressors: Exploring text classification tasks using gzip compressor

## Introduction

This repository contains the source code and documentation for my bachelor thesis project, titled "The power of compressors: Exploring text classification tasks using gzip compressor". The project aims to discover the usage of the gzip compressor based text classifier, in comparison to Standard NLP Classifier such as Decision Trees, Naive Bayes and BERT on three core datasets and three more popular text classification datasets.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Getting Started

- Download anaconda and run the python notebooks on jupyter lab.

- Download the datasets here:
  - Cancer type detection: https://www.kaggle.com/datasets/falgunipatel19/biomedical-text-publication-classification
  - E-Commerce: https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification
  - Phishing E-mails: https://www.kaggle.com/datasets/subhajournal/phishingemails
  - ACL-ARC: https://huggingface.co/datasets/hrithikpiyush/acl-arc
  - SST-2: https://huggingface.co/datasets/sst2
  - 20News: https://huggingface.co/datasets/newsgroup

### Prerequisites

Before running this project, make sure you have the following prerequisites installed:

- anaconda / python3
- pandas
- numpy
- transformers
- torch
- accellerate
- optuna
- nltk
- scikit-learn
- seaborn
- matplotlib
- wordcloud

### Installation

1. Clone this repository to your local machine:
   ```sh
   git clone https://github.com/makarwu/thesis.git
   ```
   Navigate to the project directory:
   ```
   cd thesis
   ```

### Project Structure

- `bert/`: This directory contains the BERT model approach. You can find a detailed explanation of the model in the `ecommerce_bert.ipynb` file. All other files follow the same scheme.
- `compressor/`: This directory contains the gzip compressor based text classification approach on the datasets. You can find a detailed explanation of the model in the `ecommerce_gzip.ipynb` file. All other files follow the same scheme.
- `datasets/`: This directory contains all the csv-datasets used for the thesis.
- `dt/`: This directory contains the Decision Tree text classification approach, applied on the datasets. You can find a detailed explanation of the model in the `ecommerce_dt.ipynb` file. All other files follow the same scheme.
- `dummy/`: This directory contains the baseline classifier model, applied on all datasets. You can find a detailed explanation of the model in the `ecommerce_baseline.ipynb` file. All other files follow the same scheme.
- `eda/`: This directory contains the Exploratory Data Analysis of the core datasets.
- `nb/`: This directory contains the Naive Bayes text classification approach. You can find a detailed explanation of the model in the `ecommerce_nb.ipynb` file. All other files follow the same scheme.

### Usage

- To run the code, please donwload the core and standard NLP datasets from kaggle & huggingface from the links provided above. Additionaly, store them into the datasets folder.

### Acknowledgements

This thesis was motivated from the paper “Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors (Jiang et al., Findings 2023).
