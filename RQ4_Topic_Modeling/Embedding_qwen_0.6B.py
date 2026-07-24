import pandas as pd
import glob
import os
import numpy as np
import torch
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_DIR = "new_text_body_cleaned" 

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" 

BATCH_SIZE = 32

EMBEDDINGS_FILE = "new_text_body_cleaned_embeddings.npy"
DOCUMENTS_FILE = "new_text_body_cleaned_documents.pkl"

MAX_SEQ_LENGTH = 1024

def check_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU: {device_name}")

        if torch.cuda.is_bf16_supported():
            print("GPU supports BF16 acceleration, will use bfloat16 precision.")
            return "cuda", torch.bfloat16
        else:
            print("GPU will use FP16 (half-precision) acceleration.")
            return "cuda", torch.float16
    else:
        print("No GPU detected.")
        return "cpu", torch.float32

def load_documents(input_dir: str) -> list[str]:
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No .csv files found in '{input_dir}'.")
        return []

    print(f"loading documents from {len(csv_files)} files...")
    all_docs = []
    
    for file_path in tqdm(csv_files, desc="Reading files"):
        try:
            df = pd.read_csv(file_path)
            if 'cleaned_text' in df.columns:
                docs = df['cleaned_text'].dropna().astype(str).tolist()
                docs = [d for d in docs if d.strip()] 
                all_docs.extend(docs)
            else:
                print(f"Warning: {file_path} is missing the 'cleaned_text' column.")
        except Exception as e:
            print(f"Error occurred while reading {file_path}: {e}")
            
    return all_docs

def main():
    device, torch_dtype = check_gpu()
    documents = load_documents(INPUT_DIR)
    if not documents:
        print("No valid documents loaded, terminating program.")
        return
    print(f"\nSuccessfully loaded {len(documents):,} valid documents.")
    print(f"Loading model '{MODEL_NAME}' to {device}...")
    try:
        model = SentenceTransformer(
            MODEL_NAME, 
            device=device,
            trust_remote_code=True,  
            model_kwargs={
                "torch_dtype": torch_dtype, 
                "attn_implementation": "sdpa" 
            }
        )
        
        model.max_seq_length = MAX_SEQ_LENGTH
        print(f" Model loaded successfully.")
        print(f" - Precision: {torch_dtype}")
        print(f" - Maximum sequence length: {model.max_seq_length}")
        
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        print("Hint: Please check your network connection or verify that the HuggingFace cache is not corrupted.")
        return

    print(f"\nStarting embedding of {len(documents):,} documents...")
    print(f"Batch Size: {BATCH_SIZE}")
    
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True 
    )

    print(f"Embedding completed! Matrix shape: {embeddings.shape}")
    
    try:
        print("Saving results...")
        np.save(EMBEDDINGS_FILE, embeddings)
        with open(DOCUMENTS_FILE, "wb") as f:
            pickle.dump(documents, f)
            
        print(f"Processing completed! Files saved:")
        print(f" - Embeddings: {EMBEDDINGS_FILE}")
        print(f" - Documents: {DOCUMENTS_FILE}")

    except Exception as e:
        print(f"Error occurred while saving files: {e}")

if __name__ == "__main__":
    main()