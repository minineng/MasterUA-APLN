from utils.extractor import Extractor
from utils.io import *
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def analyze_term_scores(corpus: dict, extractor: Extractor):
    corpus_content = []
    corpus_names = list(corpus.keys())
    analyzed_corpus = {}
    vectorizer = TfidfVectorizer()
    for _, path in corpus.items():
        corpus_content.append(extractor.normalize_text(read_txt(path)))
    tfidf_matrix = vectorizer.fit_transform(corpus_content)
    feature_names = vectorizer.get_feature_names_out()
    for i in range(tfidf_matrix.shape[0]):  
        doc = tfidf_matrix.getrow(i)  # Get i-th row (sparse vector)
        # Get indices & data of nonzero elements, sorted by score DESC
        top_idx = np.argsort(doc.data)[::-1]
        top_features_idx = doc.indices[top_idx]
        top_scores = doc.data[top_idx]
        
        doc_terms = {feature_names[idx]: score 
                    for idx, score in zip(top_features_idx, top_scores)}
        analyzed_corpus[corpus_names[i]] = doc_terms
    return analyzed_corpus

def run_document_term_scoring(in_folder: str, out_folder: str, extractor: Extractor):

    if not os.path.exists(in_folder):
        print(f"Error: {in_folder} not found.")
        return

    corpus = {}
    for file in os.listdir(in_folder):
        if file.endswith(".clean.md"):
            in_md = os.path.join(in_folder, file)
            if not os.path.exists(in_md):
                continue  
            filename = file.replace('.clean.md', '')     
            corpus[filename] = in_md
    
    analyzed_corpus = analyze_term_scores(corpus, extractor)
    for i, (name, features) in enumerate(analyzed_corpus.items()):
        analysis_file = os.path.join(out_folder, f"{name}_analysis.json")
        write_json(dict(features), analysis_file) 
        generate_word_cloud(analysis_file, os.path.join(out_folder, f"{name}_wordcloud.png"))
        print(f"File analyzed: {name} -> {analysis_file}")

def generate_word_cloud(json_file_path: str, output_image: str, min_score: float = 0.06, top_k: int = 80):

    data = read_json(json_file_path)

    items = [(w, float(s)) for w, s in data.items() if s > min_score]
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