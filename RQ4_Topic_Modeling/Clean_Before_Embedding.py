import pandas as pd
import re
import glob
import os

try:
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
except re.error:
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = EMOJI_PATTERN.sub(r" ", text)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`\n]+?`", " ", text)
    text = re.sub(r"^\s*at .*\n?", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\b\w+(error|exception)\b[:\s\S]*?\n", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\d+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*_~\[\]()]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def process_all_issues(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"created directory: {output_dir}")

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"{input_dir} does not contain any .csv files. Please check the directory.")
        return

    print(f"Found {len(csv_files)} .csv files. Starting processing...")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)

            title_col = 'title' if 'title' in df.columns else ('original_issue_title' if 'original_issue_title' in df.columns else None)
            body_col = 'body' if 'body' in df.columns else ('text_body' if 'text_body' in df.columns else None)

            if not title_col and not body_col:
                print(f"warning: {file_path} is missing valid title or body columns, skipping.")
                continue

            text_series = pd.Series("", index=df.index)

            if title_col:
                text_series += df[title_col].fillna("").astype(str) + " "
            
            if body_col:
                text_series += df[body_col].fillna("").astype(str)

            df['text'] = text_series.str.strip()
            
            print(f"cleaning {os.path.basename(file_path)}...")
            df['cleaned_text'] = df['text'].apply(preprocess_text)

            df['word_count'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))
            df = df[df['word_count'] >= 10].copy()

            base_filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, base_filename)

            columns_to_save_final = [col for col in df.columns if col not in ['text', 'word_count']]
            if 'cleaned_text' not in columns_to_save_final:
                columns_to_save_final.append('cleaned_text')

            df_cleaned = df[columns_to_save_final]
            df_cleaned.to_csv(output_path, index=False)

        except pd.errors.EmptyDataError:
            print(f"warning: {file_path} is empty, skipped.")
        except Exception as e:
            print(f"error occurred while processing {file_path}: {e}")

    print(f"\nprocessing completed! All cleaned files saved to {output_dir}")

if __name__ == "__main__":
    INPUT_DIRECTORY = "new_text_body" 
    OUTPUT_DIRECTORY = "new_text_body_cleaned"
    
    process_all_issues(INPUT_DIRECTORY, OUTPUT_DIRECTORY)