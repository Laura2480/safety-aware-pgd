import gc
import glob
import numpy as np
import dask.dataframe as dd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
from xgboost import XGBClassifier
import warnings


def test_col_comb_inc(df_train, df_test_out, remaining_columns, selected_columns=[]):
    df_y_train = df_train['label']
    df_y_test_out = df_test_out['label']

    y_train = torch.tensor(np.array(df_y_train.values.astype(np.int32))).to('cuda')
    y_test_out = torch.tensor(np.array(df_y_test_out.values.astype(np.int32)))
    best_composite_score = 0
    all_scores = {}
    while remaining_columns:
        scores = {}

        for col in remaining_columns:
            current_cols = selected_columns + [col]

            train_data = np.hstack([np.stack(df_train[c].apply(flatten_embedding).values) for c in current_cols])
            test_data = np.hstack([np.stack(df_test_out[c].apply(flatten_embedding).values) for c in current_cols])

            X_train_current = torch.tensor(train_data).to('cuda')
            X_test_current = torch.tensor(test_data).to('cuda')

            clf = XGBClassifier(tree_method="hist", device="cuda", n_estimators=50, random_state=42)
            clf.fit(X_train_current, y_train)
            test_preds = clf.predict(X_test_current)

            test_acc = accuracy_score(y_test_out, test_preds)

            scores[col] = (test_acc)

            print(f"Colonne {current_cols}: test_acc={test_acc:.4f}")
            del clf, X_train_current, X_test_current, test_preds, train_data, test_data
            gc.collect()
            torch.cuda.empty_cache()

        best_candidate = max(scores, key=scores.get)
        if scores[best_candidate] > best_composite_score:
            selected_columns.append(best_candidate)
            remaining_columns.remove(best_candidate)
            best_composite_score = scores[best_candidate]
            print(f"Aggiunta la colonna {best_candidate} con score = {best_composite_score:.4f}")
        else:
            print("Nessun miglioramento trovato, interrompo la selezione.")
            break

        gc.collect()
        torch.cuda.empty_cache()
    return best_candidate
def flatten_embedding(embedding):
    return np.array(embedding).flatten()

def load_parquets_df(base, base_cols, parts=1, only_specific=False, specific_columns=None):
    warnings.filterwarnings("ignore")
    files = sorted(glob.glob(f"{base}*.parquet"))

    sample_ddf = dd.read_parquet(files[0])
    all_cols = sample_ddf.columns.tolist()
    len_heads = int((len(all_cols) - len(base_cols)) / parts)
    embedding_cols = [c for c in all_cols if c not in base_cols]
    if not only_specific:
        chunk_size = len_heads
        embedding_cols_chunks = [
            embedding_cols[i:i + chunk_size]
            for i in range(0, len(embedding_cols), chunk_size)
        ]

        print(f"\n Sono disponibili {len(embedding_cols_chunks)} chunk.")
        for i, chunk in enumerate(embedding_cols_chunks):
            print(f" Chunk {i + 1}: {chunk[:3]}... (tot={len(chunk)})")

        selected_chunk_idx = int(
            input("\n🔢 Inserisci il numero del chunk da caricare (1-{}): ".format(len(embedding_cols_chunks)))) - 1

        if selected_chunk_idx < 0 or selected_chunk_idx >= len(embedding_cols_chunks):
            print(" Numero non valido! Uscita...")
            exit()
        specific_columns = embedding_cols_chunks[selected_chunk_idx]
        cols_to_load = base_cols + specific_columns

        print(
            f"\n Caricamento del Chunk {selected_chunk_idx + 1} con {len(cols_to_load) - len(base_cols)} colonne embedding.")
    else:
        specific_columns = specific_columns if specific_columns else embedding_cols
        cols_to_load = base_cols + specific_columns

    dfs = []

    for f in tqdm(files, desc='Caricamento'):
        ddf_chunk = dd.read_parquet(f, columns=cols_to_load)
        pdf_chunk = ddf_chunk.compute()  # Pandas DataFrame
        dfs.append(pdf_chunk)

    print("\nCaricamento ed elaborazione completati!")

    # Concatena tutti in un unico DataFrame
    df = pd.concat(dfs, ignore_index=True)
    embedding_cols = cols_to_load
    del dfs  # rimuovi l'intera lista di DataFrame
    import gc
    gc.collect()
    print("Numero totale di righe:", len(df))
    return df, specific_columns

