from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from AITADataset import create_data_loader
from SentimentClassifier import SentimentClassifier


# Training helper method
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader, desc="train", leave=False):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

# Evaluation helper method
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in tqdm(data_loader, desc="eval", leave=False):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

RANDOM_SEED = 7
BATCH_SIZE = 7
MAX_LEN = 512 # Could be optimized
EPOCHS = 10

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device", device)

    # Use Pandas to read dataset
    df = pd.read_csv(Path(__file__).parent / "../data/aita_clean.csv")
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

    # Import bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split train, test, validation
    df_train, df_test = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_SEED
    )

    df_val, df_test = train_test_split(
        df_test,
        test_size=0.5,
        random_state=RANDOM_SEED
    )

    print(df_train.shape, df_test.shape, df_val.shape)


    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    model = SentimentClassifier(n_classes=2).to(device)

    history = defaultdict(list)
    best_accuracy = 0

    loss_fn = nn.CrossEntropyLoss().to(device)
    # we can't use pytorch.optim.AdamW because it won't let us disable bias correction
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for _ in tqdm(range(EPOCHS), desc="epoch"):
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print("\n")
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

if __name__ == "__main__":
    main()
