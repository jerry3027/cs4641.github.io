import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use Pandas to read dataset
df = pd.read_csv("./project1/data/aita_clean.csv")
# print(df.head())
# print(df.shape)

# Check for class imbalance
# sns.countplot(df.is_asshole)
# plt.xlabel('is asshole')
# plt.show() # Data is very imbalanced, we need to modify the data

# Preprocess data
# Make data balanced (26500 of each label)
df = pd.concat([df[df['is_asshole']==1].sample(n=26500), df[df['is_asshole']==0].sample(n=26500)])
# Make everything lower case
df['title'] = df['title'].str.lower()
df['body'] = df['body'].str.lower()
# Merge title and body
df['text'] = df['title'] + ' ' + df['body']
df['text'] = df['text'].str.replace('\n', '')

# Import bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

