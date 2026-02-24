from utils.extractor import Extractor
from utils.io import *
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import json

def analyze_term_scores(input_file: str, output_file: str, extractor: Extractor):
    normalized_text = extractor.normalize_text(read_txt(input_file))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([normalized_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_scores = dict(zip(feature_names, [float(f"{score:.4f}") for score in tfidf_scores]))
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    write_json(dict(sorted_words), output_file) 

def run_document_term_scoring(in_folder: str, out_folder: str, extractor: Extractor):

    if not os.path.exists(in_folder):
        print(f"Error: {in_folder} not found.")
        return

    for folder in os.listdir(in_folder):
        folder_path = os.path.join(in_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        in_md = os.path.join(folder_path, "text.md")
        if not os.path.exists(in_md):
            continue       

        analysis_file = os.path.join(out_folder, f"{folder}_analysis.json")
        analyze_term_scores(in_md, analysis_file, extractor)
        generate_word_cloud(analysis_file, os.path.join(out_folder, f"{folder}_wordcloud.png"))
        print(f"File analyzed: {folder} -> {analysis_file}")

def generate_word_cloud(json_file_path: str, output_image: str, min_score: float = 0.06, top_k: int = 80):

    data = read_json(json_file_path)

    items = [(w, float(s)) for w, s in data.items()]
    items = [(w, s) for w, s in items if s > min_score]
    items.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        items = items[:top_k]

    if not items:
        print(f"Warning: no terms above min_score={min_score} in {json_file_path}")
        return

    freq = dict(items)

    wc = WordCloud(
        width=800,
        height=800,
        background_color="white",
        min_font_size=10
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_image)
    plt.close()