# Gen-Z Slang Translator

This project demonstrates how to fine-tune a GPT-2 model to translate formal English sentences into Gen-Z slang. It utilizes a dataset of formal-to-slang sentence pairs to teach the model context and style transformation.

## How to Use 

### 1. Environment Setup
Ensure you are running this notebook in a Google Colab environment with GPU acceleration enabled. The necessary libraries are already imported in the first code cell.

```python
import os
import time
import datetime
from google.colab import drive

import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt
%matplotlib inline

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

import nltk
nltk.download('punkt_tab')
```

Verify GPU availability by running the `!nvidia-smi` command:

```python
!nvidia-smi
```

### 2. Data Loading and Preparation
The dataset `genz_dataset.csv` is loaded from a Hugging Face repository. Grant access to your Hugging Face token to load the dataset or download the .csv file and provide your local path to the data folder.


### 3. Model Initialization and Training
Run the notebook to initialize the model and begin training.

### 4. Translation and Generation
To translate a formal sentence, edit `formal_sentence`. Do not edit the prompt. The model can also be altered slighlty to translate slang to formal. To do this, edit `prompt` as follows:

```python
slang_sentence = "I'm drained fr"
prompt = f"<|startoftext|> <Slang:> {slang_sentence} <Translate> <Formal:>"
```

### 5. Saving and Loading the Model
The trained model and tokenizer can be saved to a specified directory and reloaded later for inference without retraining.

```python
output_dir = './model_save/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

To load the model:
```python
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model.to(device)
```
