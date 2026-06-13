# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# Tokenize sentence pairs
def tokenize_data(texts):
    '''
    This function takes a pandas DataFrame containing two columns: 
        - 'miRNA/isomiR': sequences of miRNAs or isomiRs 
        - 'mRNA': corresponding mRNA target site sequences
    
    It returns a dictionary of tokenized outputs formatted as PyTorch tensors:
        - input_ids: numerical IDs for tokens
        - token_type_ids: distinguishes miRNA/isomiR sequence (0) from mRNA sequence (1)
        - attention_mask: 1 for real tokens, 0 for padding
    '''
    return tokenizer(
        list(texts['miRNA/isomiR']),
        list(texts['mRNA']),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

      
      
# Custom Dataset class for miRNA/isomiR - mRNA interaction data
class miRNAMRNADataset(Dataset):
  def __init__(self, encodings, labels):
      '''
        encodings: tokenized input pairs (dictionary containing input_ids, token_type_ids, attention_mask)
        labels: interaction labels (0 or 1) for each input pair; 1 = positive interaction; 0 = negative interaction

      '''
      self.encodings = encodings
      self.labels = labels
  
  def __len__(self):
      # Returns the total number of samples in the dataset
      return len(self.labels)
  
  def __getitem__(self, idx):
      # Fetch a single data sample by index
      item = {key: val[idx].clone().detach().to(device) for key, val in self.encodings.items()}
      
      # Add the corresponding label for the fetched sample
      item["labels"] = self.labels[idx].clone().detach().to(device)

      # Returns the final dictionary containing both inputs and label
      return item
  

def compute_metrics(eval_pred):
    '''
        This function takes the model's raw predictions and true labels, then calculates:
        - Accuracy
        - F1 score
        - Precision
        - Recall
        - AUROC (Area Under ROC Curve)

        Returns a dictionary of these metrics.
    '''
    predictions, labels, metrics = eval_pred
    
    # Convert raw logits into probabilities.
    probs = torch.sigmoid(torch.tensor(predictions[:, 1])).numpy()
    
    # Convert probabilities into binary predictions (0 or 1) using threshold = 0.5
    binary_preds = (probs > 0.5).astype(int)
    
    # Create DataFrame
    df_results = pd.DataFrame({
      "True_Label": labels,
      "Predicted_Label": binary_preds,
      "Probability": probs
    })
    
    #df_results.to_csv("/lustre/fs1/home/ro050617/mirna_isomir_prediction/codes/outputs/bert/five_folds/fold1/predictions/predictions.csv", index=False)
    
    accuracy = accuracy_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds)
    recall = recall_score(labels, binary_preds)
    auroc = float(roc_auc_score(labels, probs))
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auroc": auroc
    }
          
          
if __name__ == '__main__':


  # Check if GPU is available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # change working directory to script's folder
  script_dir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(script_dir)  

  # Load datasets
  print("Dataset Loading.......")
  train_df = pd.read_csv("sample_train_df.csv")
  test_df = pd.read_csv("sample_test_df.csv")
  
  print("Training data: {}".format(train_df.shape[0]))
  print("Test data: {}".format(test_df.shape[0]))    

  ##Train Split
  print("Train Split......")
  train_texts = train_df[['miRNA/isomiR', 'mRNA']]
  train_labels = train_df['label']
  
  # initialize the tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
  print("Tokenizing...........")
  train_encodings = tokenize_data(train_texts)
  test_encodings = tokenize_data(test_df[['miRNA/isomiR', 'mRNA']])  # Tokenize independent test data

  # Convert labels to tensors
  train_labels = torch.tensor(train_labels.values)
  test_labels = torch.tensor(test_df['label'].values)  # Independent test labels
    
  # Create pytorch datasets
  train_dataset = miRNAMRNADataset(train_encodings, train_labels)
  test_dataset = miRNAMRNADataset(test_encodings, test_labels)
  
  # Load pre-trained BERT model and move to GPU
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
  
  # Training arguments
  training_args = TrainingArguments(
      eval_strategy="no",               # No evaluation during training. Set yes if you have validation dataset.
      save_strategy="no",               # Not saving the checkpoints. Set yes if you want to save.
      per_device_train_batch_size=64,   # Training data batch size
      per_device_eval_batch_size=64,    # Evaluation data batch size
      num_train_epochs=1,
      weight_decay=0.01,                # Regularization to prevent overfitting
      dataloader_pin_memory=False,      # Set True for faster GPU transfer
      logging_dir="./logs",             # Directory for logs
      logging_strategy = "epoch",       # Logging Frequency
      report_to = "none",               # Not reporting to external tools such as WandB
      load_best_model_at_end=False,     # Set True if you want to Load best model at end of training
      disable_tqdm = True               # Disabling the progress bar
  )
  
  # Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
  )
  
  print("Training..............")
  # Train model
  trainer.train()  # Resume training if interrupted

  print("Evaluating.............")  
  predictions = trainer.predict(test_dataset)
  print("Test Result: ")
  print(compute_metrics(predictions))

