# Define ModerationModel as before
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from open_clip.transformer import ResidualAttentionBlock,LayerNorm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

class ModerationDatasetSequence(Dataset):
    def __init__(self, data, seq_len, embedding_dim, num_labels=11):
        """
        Parametri:
        - data: array/tensor con shape (num_samples, seq_len * embedding_dim + num_labels)
        - seq_len: lunghezza della sequenza (numero di embeddings per esempio)
        - embedding_dim: dimensione di ciascun embedding
        - num_labels: numero di label (le ultime colonne di d ata)
        """
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        total_features = seq_len * embedding_dim
        # Prendi le prime total_features e risagomale in (seq_len, embedding_dim)
        self.embeddings = data[:, :total_features].reshape(-1, seq_len, embedding_dim)
        # Le ultime num_labels colonne contengono le label
        self.labels = data[:, total_features:]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return embeddings, labels


# --- Definizione del modello MLP ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ---------------------------
# DEFINIZIONE DEL MODELLO CON OUTPUT GREZZO (LOGITS)
# ---------------------------
class ModerationHeadMLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_layer=3,scale_factor=0.5,input_scale_factor=2/3, dropout_rate=0.3):
        """
        input_dim: dimensione dell'input
        hidden_layer: numero di blocchi hidden
        Il modello restituisce i logits grezzi; l'applicazione della funzione sigmoide deve essere effettuata esternamente.
        """
        super(ModerationHeadMLP, self).__init__()
        factor = scale_factor

        # Normalizzazione dell'input
        self.ln0 = nn.LayerNorm(input_dim)
        # Primo layer fully connected per proiettare l'input
        hidden_dim = int(input_dim * input_scale_factor)
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Creazione dinamica dei blocchi hidden
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_layer):
            block = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, int(hidden_dim * factor))
            )
            self.hidden_layers.append(block)
            hidden_dim = int(hidden_dim * factor)

        # Layer di output: dimensione 1 per la classificazione binaria
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.ln0(x)
        x = self.input_fc(x)
        for layer in self.hidden_layers:
            x = layer(x)
        # Restituiamo i logits grezzi senza applicare la sigmoide
        logits = self.fc_out(x)
        return logits

# ---------------------------
# FUNZIONI DI TRAINING E VALUTAZIONE
# ---------------------------

def stratified_split(df, strat_cols, test_proportion=0.1):
    """
    Suddivide un DataFrame in train e test set mantenendo la distribuzione delle classi
    secondo le colonne di stratificazione specificate.

    Parametri:
    - df: pd.DataFrame - il DataFrame da suddividere
    - strat_cols: list - lista delle colonne su cui stratificare
    - test_proportion: float - proporzione del test set (default 0.1)

    Ritorna:
    - df_train_in: pd.DataFrame - train set
    - df_test_in: pd.DataFrame - test set
    """
    df = df.copy()

    # Crea chiave di stratificazione
    df['strat_key'] = df[strat_cols].astype(str).agg('_'.join, axis=1)

    # Calcola numero di gruppi
    n_groups = df['strat_key'].nunique()

    # Numero desiderato di esempi per gruppo
    desired_total = int(len(df) * test_proportion)
    desired_per_group = int(np.ceil(desired_total / n_groups))

    # Campionamento stratificato
    df_test_in = df.groupby('strat_key', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), desired_per_group), random_state=42)
    ).reset_index(drop=True)

    # Rimozione degli esempi test dal train
    df_train_in = df.drop(df_test_in.index).drop(columns=['strat_key'])
    df_test_in = df_test_in.drop(columns=['strat_key'])

    return df_train_in, df_test_in


def train_model_with_best(model, train_loader, test_loaders, num_epochs=10, lr=1e-3, device='cuda', save_path='best_model.pth'):
    """
    Addestra il modello e ad ogni epoca valuta la somma delle accuracy sui test loader.
    Salva il best model in base alla somma delle accuracy.
    Utilizziamo la BCEWithLogitsLoss che applica internamente la sigmoide.
    """
    model.to(device)
    # BCEWithLogitsLoss applica la sigmoide internamente per maggiore stabilità numerica
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_total_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device).float()
            optimizer.zero_grad()
            logits = model(batch_data)  # logits grezzi
            loss = criterion(logits, batch_labels.unsqueeze(1))  # le etichette devono avere dimensione (batch, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Valutazione su tutti i test loader
        accuracies = []
        total_acc = 0.0

        for test_loader in test_loaders:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits = model(data)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long().view(-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            accuracies.append(acc)
            total_acc += acc

        mean_acc = total_acc / len(test_loaders)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracies: {[f'{a:.4f}' for a in accuracies]} - Total Accuracy: {total_acc:.4f}")

        # Salva il modello se la somma totale delle accuracy migliora
        if total_acc > best_total_acc:
            best_total_acc = total_acc
            torch.save(model.state_dict(), save_path)
            print(f"Nuovo best model salvato con Total Accuracy Sum: {best_total_acc:.4f}")

    return model


def evaluate_metrics(model, data_loader,th, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            # Applichiamo la sigmoide esternamente
            probs = torch.sigmoid(logits)
            preds = (probs > th).long().view(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec

def stratified_sample(df_safe_orig, n_samples, group_col="OLD_TAXONOMY"):
    # Raggruppa i safe per il valore originale (prima che vengano convertiti in "SAFE")
    groups = df_safe_orig.groupby(group_col)
    n_groups = len(groups)
    # Numero minimo da prelevare per ogni gruppo
    n_per_group = n_samples // n_groups
    samples_list = []
    # Per ogni gruppo, campiona n_per_group; se il gruppo ha meno campioni, ne prende tutti
    for grp_val, grp_df in groups:
        if len(grp_df) >= n_per_group:
            samples_list.append(grp_df.sample(n=n_per_group, random_state=42))
        else:
            samples_list.append(grp_df)
    # Se non abbiamo raggiunto n_samples, aggiungiamo campioni casuali dagli stessi gruppi
    df_samples = pd.concat(samples_list)
    if len(df_samples) < n_samples:
        n_extra = n_samples - len(df_samples)
        extra_samples = df_safe_orig.drop(df_samples.index).sample(n=n_extra, random_state=42)
        df_samples = pd.concat([df_samples, extra_samples])
    return df_samples

# Consideriamo i campioni in cui la label è SAFE o uguale a target_val.
def subset_for_target(df, X_all,label_col,label_mapping,target_val):
    mask = df[label_col].isin([label_mapping["SAFE"], target_val])
    y_bin = df.loc[mask, label_col].apply(lambda x: 1 if x == target_val else 0).values
    X_subset = X_all[mask]
    return y_bin, X_subset