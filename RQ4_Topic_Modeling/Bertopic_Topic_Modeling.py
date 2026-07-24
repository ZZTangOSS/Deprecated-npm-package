import numpy as np
import pickle
import pandas as pd
import os
import torch
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text # 补充引入 text 用于停用词
from joblib import Memory
from sentence_transformers import SentenceTransformer
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" 
EMBEDDINGS_FILE = "new_text_body_cleaned_embeddings.npy"
DOCUMENTS_FILE = "new_text_body_cleaned_documents.pkl"

MODEL_SAVE_PATH = "new_text_body_cleaned_bertopic_qwen_keybert.safetensors"
RESULTS_DIR = "new_text_body_cleaned_bertopic_results_keybert"
VISUALIZATIONS_DIR = "new_text_body_cleaned_bertopic_visualizations_keybert"
JOBLIB_CACHE_DIR = "new_text_body_cleaned_joblib_cache"
MIN_TOPIC_SIZE = 60 

for dir_path in [RESULTS_DIR, VISUALIZATIONS_DIR, JOBLIB_CACHE_DIR]: 
    os.makedirs(dir_path, exist_ok=True)

safe_cache_path = os.path.abspath(JOBLIB_CACHE_DIR)
os.environ["JOBLIB_TEMP_FOLDER"] = safe_cache_path
joblib_memory = Memory(location=safe_cache_path, verbose=0)

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        else:
            return "cuda", torch.float16
    return "cpu", torch.float32

def main():
    try:
        print(f"loading '{EMBEDDINGS_FILE}'...")
        embeddings = np.load(EMBEDDINGS_FILE)
        print(f"loading '{DOCUMENTS_FILE}'...")
        with open(DOCUMENTS_FILE, "rb") as f:
            documents = pickle.load(f)
    except FileNotFoundError:
        print(f"error: '{EMBEDDINGS_FILE}' or '{DOCUMENTS_FILE}' not found. Please ensure the files exist.")
        return

    valid_indices = []
    valid_documents = []
    for i, doc in enumerate(documents):
        if isinstance(doc, str) and doc.strip():
            valid_indices.append(i)
            valid_documents.append(doc)

    if len(valid_documents) < len(documents):
        print(f"filter {len(documents) - len(valid_documents)} invalid documents. Updating embeddings accordingly.")
        embeddings = embeddings[valid_indices, :]
        documents = valid_documents
    else:
        print("Document integrity check passed, no filtering needed.")

    print(f"valid documents: {len(documents):,} | embedding matrix shape: {embeddings.shape}")

    device, torch_dtype = check_gpu()
    try:
        embedding_model = SentenceTransformer(
            MODEL_NAME, 
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch_dtype, "attn_implementation": "sdpa"}
        )
        embedding_model.max_seq_length = 1024 
        print("Qwen loaded successfully.")
    except Exception as e:
        print(f"Qwen loading error: {e}")
        return

    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0,
        metric='cosine', 
        random_state=42, 
        verbose=True 
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE, 
        metric='euclidean',
        cluster_selection_method='eom', 
        prediction_data=True,
        memory=joblib_memory
    )

    github_stop_words = list(text.ENGLISH_STOP_WORDS)
    custom_words = ["github", "issue", "comment", "problem", "repository", "nan", "null"] 
    github_stop_words = list(set(github_stop_words + custom_words))

    vectorizer_model = CountVectorizer(
        stop_words=github_stop_words, 
        ngram_range=(1, 3), 
        min_df=5, 
        max_df=0.90
    )
    
    keybert_model = KeyBERTInspired()

    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    
    representation_model = {
        "Main": keybert_model,
        "Diversity": mmr_model
    }

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model, 
        embedding_model=embedding_model, 
        language="english", 
        verbose=True,
        calculate_probabilities=False 
    )

    topics, probs = topic_model.fit_transform(documents, embeddings)
    print("completed")

    topic_model.save(MODEL_SAVE_PATH, serialization="safetensors", save_embedding_model=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(RESULTS_DIR, "topic_summary_keybert.csv"), index=False, encoding='utf-8-sig')
    
    out_df = pd.DataFrame({'document': documents, 'topic': topics})
    out_df.to_csv(os.path.join(RESULTS_DIR, "document_assignments.csv"), index=False, encoding='utf-8-sig')
    print(f"saved results to: {RESULTS_DIR}")

    try:
        fig = topic_model.visualize_barchart(top_n_topics=15, n_words=8)
        fig.write_html(os.path.join(VISUALIZATIONS_DIR, "topic_barchart_keybert.html"))

        print("calculating 2D UMAP for document visualization")
        umap_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        fig_doc = topic_model.visualize_documents(documents, reduced_embeddings=umap_2d, hide_document_hover=True)
        fig_doc.write_html(os.path.join(VISUALIZATIONS_DIR, "document_scatter.html"))
        
        print("completed visualizations, saved to:", VISUALIZATIONS_DIR)
    except Exception as e: 
        print(f"error: {e}")

if __name__ == "__main__":
    main()