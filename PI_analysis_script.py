import pandas as pd
from transformers import BertTokenizer
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

# Define the consensus sequence for the protease inhibitor
protease_consensus = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

# List of drug columns in the dataset
drug_cols = ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"]

# Define drug-specific resistance thresholds (fold change)
# (References: Pironti et al., JAIDS 2017; Shen et al., 2016)
drug_thresholds = {
    "FPV": 4.0,
    "ATV": 3.0,
    "IDV": 3.0,
    "LPV": 9.0,
    "NFV": 3.0,
    "SQV": 3.0,
    "TPV": 2.0,
    "DRV": 10.0
}

#Load the dataset from the txt file we downloaded
df = pd.read_csv("PI_dataset.txt", sep="\t")

# Verify that the sequence columns exist.
# We'll filter columns whose names start with 'P' and are followed by digits.
# This assumes that the sequence columns are named 'P1', 'P2', ..., 'P99'.
p_columns = [col for col in df.columns if col.startswith("P") and col[1:].isdigit()]

# Sort the columns in numerical order (P1, P2, ..., P99)
p_columns = sorted(p_columns, key=lambda x: int(x[1:]))

# Check that we have exactly 99 positions (P1 to P99)
# This should be the case as protease is a homodimer with each subunit having 99 amino acids 
if len(p_columns) != 99:
    print(f"Warning: Expected 99 sequence positions but found {len(p_columns)}")

def concatenate_sequence(row, columns, consensus_seq):
    """
    Concatenate amino acid columns into a single sequence string.
    Handling missing/ambiguous symbols (from the dataset description):
      - If a cell is NaN, replace with 'X' (unknown).
      - If the value is '.', replace with 'X' (unknown, no sequence data).
      - If the value is '-', replace with the corresponding consensus residue.
    """
    seq_list = []
    # Enumerate over the sorted columns so we know the position (0-indexed)
    for idx, col in enumerate(columns):
        aa = row[col]
        if pd.isna(aa):
            # If the value is NaN, mark as unknown
            aa = 'X'
        else:
            aa = str(aa).strip()
            # Replace '.' with unknown, and '-' with the consensus residue
            if aa == '.':
                aa = 'X'
            elif aa == '-':
                aa = consensus_seq[idx]
        seq_list.append(aa)
    
    # Join the amino acids into a continuous string (without spaces)
    raw_seq = ''.join(seq_list)
    
    # Insert spaces between each amino acid for ProteinBERT tokenization
    formatted_seq = " ".join(list(raw_seq))
    
    return formatted_seq

# Apply the function to each row to create a new column with the formatted sequence
df["FormattedSequence"] = df.apply(lambda row: concatenate_sequence(row, p_columns, protease_consensus), axis=1)

# Display a sample formatted sequence
print("Sample Formatted Sequence:")
print(df.loc[0, "FormattedSequence"])

# Create binary labels for each drug: 1 if fold change >= threshold (resistant), else 0 (susceptible)
for drug in drug_cols:
    df[f"{drug}_label"] = df[drug].apply(lambda x: 1 if x >= drug_thresholds[drug] else 0)

# Create a combined label column as an 8-dimensional vector (order: FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV)
label_cols = [f"{drug}_label" for drug in drug_cols]
labels = df[label_cols].values  # shape: (num_samples, 8)

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
sample_encoding = tokenizer(
    df.loc[0, "FormattedSequence"],
    max_length=128,  # Adjust this based on your sequence length plus special tokens
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)
print(sample_encoding.input_ids.shape)


class HIVMultiLabelDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=128):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]  # label is an array of length 8
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # Squeeze the batch dimension for each tensor.
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        # Convert label to tensor (float for BCEWithLogitsLoss)
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

# Prepare the list of sequences from "FormattedSequence"
sequences = df["FormattedSequence"].tolist()

# ================================
# 4. DEFINE THE MULTI-LABEL CLASSIFIER MODEL
# ================================
# We define a custom classifier that uses the ProtBERT model as the base.
# The classifier will have 8 outputs (one for each drug), and we use BCEWithLogitsLoss.
class ProteinBERTMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels=8, dropout_prob=0.1):
        super(ProteinBERTMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        self.dropout = nn.Dropout(dropout_prob)
        # The hidden size of ProtBERT is typically 1024 (check model info)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output (corresponds to [CLS] token representation)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ================================
# 5. TRAINING WITH CROSS-VALIDATION
# ================================

# Training parameters
num_epochs = 25
batch_size = 16
learning_rate = 2e-5
max_length = 128  # Adjust as needed (should be >= number of tokens in formatted sequence)

# Set device 
# Testing this on my mac first, then will run on MBI server 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif torch.backends.mps.is_available():
    #This for macs with apple silicon chips
    print("Using 'mps' (multi-process service) device")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")

# Prepare multi-label labels array
multi_labels = labels  # shape (N, 8)

# Convert sequences and labels to lists for dataset creation
dataset_sequences = sequences
dataset_labels = multi_labels

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_acc_list = []
fold_f1_list = []

# Function to compute multi-label accuracy (exact match) and F1 score (macro)
def compute_metrics(true_labels, pred_probs, threshold=0.5):
    pred_labels = (pred_probs >= threshold).astype(int)
    # Exact match accuracy (all labels correct)
    exact_match_acc = np.mean(np.all(pred_labels == true_labels, axis=1))
    # Compute macro F1 (average F1 over labels)
    f1s = []
    for i in range(true_labels.shape[1]):
        f1s.append(f1_score(true_labels[:, i], pred_labels[:, i], zero_division=0))
    macro_f1 = np.mean(f1s)
    return exact_match_acc, macro_f1

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_sequences)):
    print(f"\nFold {fold+1} / 5")
    
    train_seqs = [dataset_sequences[i] for i in train_idx]
    train_lbls = dataset_labels[train_idx]
    val_seqs = [dataset_sequences[i] for i in val_idx]
    val_lbls = dataset_labels[val_idx]
    
    train_dataset = HIVMultiLabelDataset(train_seqs, train_lbls, tokenizer, max_length=max_length)
    val_dataset = HIVMultiLabelDataset(val_seqs, val_lbls, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model for this fold
    model = ProteinBERTMultiLabelClassifier(num_labels=8)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop for the current fold
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on the validation set at the end of this epoch
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].cpu().numpy()
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(logits).cpu().numpy()  # get probabilities
                all_preds.append(preds)
                all_trues.append(labels_batch)
        all_preds = np.vstack(all_preds)
        all_trues = np.vstack(all_trues)
        
        epoch_acc, epoch_f1 = compute_metrics(all_trues, all_preds, threshold=0.5)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}, Val Exact Match Acc: {epoch_acc:.4f}, Val Macro F1: {epoch_f1:.4f}")
    
    # Evaluation on validation set
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].cpu().numpy()
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()  # get probabilities
            all_preds.append(preds)
            all_trues.append(labels_batch)
    
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    
    fold_acc, fold_f1 = compute_metrics(all_trues, all_preds, threshold=0.5)
    print(f"Fold {fold+1} Exact Match Accuracy: {fold_acc:.4f}, Macro F1: {fold_f1:.4f}")
    fold_acc_list.append(fold_acc)
    fold_f1_list.append(fold_f1)
    
    # Clean up
    del model
    torch.cuda.empty_cache()

# Overall results
print("\nCross-validation Results:")
print(f"Average Exact Match Accuracy: {np.mean(fold_acc_list):.4f} ± {np.std(fold_acc_list):.4f}")
print(f"Average Macro F1 Score: {np.mean(fold_f1_list):.4f} ± {np.std(fold_f1_list):.4f}")