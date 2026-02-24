import os
import spacy
import re
from transformers import pipeline
from utils.io import read_txt, read_json
import time

# SpaCy Model Loading
# We use the Spanish model to match the BOE's language.

class Extractor():

    QUESTION_CONFIG_PATH = "data/config/questions.json"
    RE_CUANTIA = re.compile(r"\d+,?\d*")

    class Tokenizer:
        CUSTOM_STOPWORDS = {"artículo", "capítulo", "resolución", "becas", "beca"}
        def __init__(self, nlp, custom_stops = []):
            self.nlp = nlp
            self.stopwords = self.CUSTOM_STOPWORDS
            self.stopwords.union(custom_stops)
            
        def __call__(self, text):
            doc = self.nlp(text)
            tokens = [
                token.lower_.strip()
                for token in doc
                if not (token.is_punct or token.is_space or token.is_stop or len(token.text) < 2)
                and token.text not in self.stopwords
            ]
            return tokens
    def __init__(self):
        print("Extractor created...")
        self.nlp = None
        self.qa_model = None
        self._initialize()
        self.question_config = read_json(self.QUESTION_CONFIG_PATH)

    def _initialize(self):
        print("Initializing NLP models...")
        try:
            self.nlp = spacy.load("es_core_news_sm", disable=["parser", "tagger", "lemmatizer", "attribute_ruler", "senter"])
            self.nlp.max_length = 2_000_000
        except:
            spacy.cli.download("es_core_news_sm")
            self.nlp = spacy.load("es_core_news_sm", disable=["parser", "tagger", "lemmatizer", "attribute_ruler", "senter"])
            self.nlp.max_length = 2_000_000

        # Question Answering
        self.qa_model = pipeline(
            "question-answering",
            model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
        )
    
    def get_tokenizer(self, custom_stops = []):
        return self.Tokenizer(self.nlp, custom_stops)

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        tokenizer = self.get_tokenizer()
        return " ".join(tokenizer(text))

    def extract_scholarship_data(self, file_path):

        starting_time = time.time()
        
        # We combine NER and QA
        if not os.path.exists(file_path):
            return None

        content = read_txt(file_path)

        # NER for fixed entities
        doc = self.nlp(content)
        issuing_body = "Not identified"
        for ent in doc.ents:
            if ent.label_ == "ORG":
                issuing_body = ent.text
                break

        question_config = self.question_config.copy()

        # QA for variable data
        questions = question_config["questions"].copy()

        results = {
            "nombre_documento": os.path.basename(file_path),
            "cuerpo_emisor": issuing_body
        }

        for key, q in questions.items():
            res = self.qa_model(question=q, context=content)
            answer = res['answer']

            if key.find("cuantia") != -1:            
                numbers = re.findall(self.RE_CUANTIA, answer.replace('.', ''))
                if numbers:
                    answer = float(numbers[0].replace(',', '.'))

            results[key] = answer

        print(f"OK - Extraction completed in {time.time() - starting_time:.2f} seconds.")
        return results