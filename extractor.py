import os
import spacy
from transformers import pipeline

# SpaCy Model Loading
# We use the Spanish model to match the BOE's language.
try:
    nlp = spacy.load("es_core_news_sm")
except:
    spacy.cli.download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# Question Answering
qa_model = pipeline(
    "question-answering",
    model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
)

def extract_scholarship_data(file_path):
    # We combine NER and QA
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        # We read the full content
        content = f.read()

    # NER for fixed entities
    doc = nlp(content[:20000])
    issuing_body = "Not identified"
    for ent in doc.ents:
        if ent.label_ == "ORG":
            issuing_body = ent.text
            break

    # QA for variable data
    questions = {
        "academic_year": "¿A qué curso académico pertenece esta convocatoria?",
        "income_linked": "¿Cuál es la cuantía de la beca fija ligada a la renta?",
        "residence_linked": "¿Cuál es la cuantía de la beca fija ligada a la residencia?",
        "deadline": "¿Cuál es la fecha límite para presentar la solicitud?"
    }

    results = {
        "document_name": os.path.basename(os.path.dirname(file_path)),
        "issuing_body": issuing_body,
        "academic_year": "",
        "aid_amounts": {
            "income_linked": "",
            "residence_linked": ""
        },
        "application_deadline": ""
    }

    # We iterate through the questions and extract the exact span from the text.
    for key, q in questions.items():
        # The model returns the 'answer' string found in the context.
        res = qa_model(question=q, context=content)
        answer = res['answer']

        if key == "academic_year":
            results["academic_year"] = answer
        elif key == "income_linked":
            results["aid_amounts"]["income_linked"] = answer
        elif key == "residence_linked":
            results["aid_amounts"]["residence_linked"] = answer
        elif key == "deadline":
            results["application_deadline"] = answer

    return results