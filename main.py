import os
from utils.preprocessing import run_preprocessing, run_markdown_cleaning
from utils.extractor import Extractor
from utils.io import *
from utils.analysis import run_document_term_scoring
from utils.summarization import SummaryGenerator

DATA_DIR = "data"
CONFIG_DIR = os.path.join(DATA_DIR,"config")
CORPUS_DIR = os.path.join(DATA_DIR,"corpus")
PREPROCESSED_DIR = os.path.join(DATA_DIR,"preprocessed")
PROCESSED_DIR = os.path.join(DATA_DIR,"processed")
ADDITIONAL_ANALYSIS_DIR = os.path.join(DATA_DIR,"analysis")
SUMMARY_DIR = os.path.join(DATA_DIR,"summary")

DEFAULT_INFORMATION_EXTRACTION_OUTPUT = "final_scholarship_dataset"

def information_extraction():
    # 1. PDF to Markdown
    # Using marker-pdf to convert original PDFs into clean Markdown
    print("--- STAGE 1: PDF PREPROCESSING ---")
    run_preprocessing(CORPUS_DIR, PREPROCESSED_DIR)
    run_markdown_cleaning(PREPROCESSED_DIR)

    question_config_path = os.path.join(CONFIG_DIR, "questions.json")
    extractor = Extractor(question_config_path)
    # 1.5 Term Scoring Analysis (Optional)
    # This step is not strictly necessary for the final dataset, but we wanted to do it to see the insights it can provide.
    run_document_term_scoring(PREPROCESSED_DIR, ADDITIONAL_ANALYSIS_DIR, extractor)
    
    # 2. Information Extraction Stage
    # This stage uses Hugging Face QA and SpaCy NER
    print("\n--- STAGE 2: NLU DATA EXTRACTION (HF + SpaCy) ---")
    final_results = []

    if not os.path.exists(PREPROCESSED_DIR):
        print(f"Error: {PREPROCESSED_DIR} folder not found. Preprocessing might have failed.")
        return

    # Navigate through the markdown created by pymupdf4llm
    for file in os.listdir(PREPROCESSED_DIR):
        if file.endswith(".clean.md"):
            try:
                extracted_data = extractor.extract_scholarship_data(os.path.join(PREPROCESSED_DIR,file))
                if extracted_data:
                    final_results.append(extracted_data)
            except Exception as e:
                print(f"Error in NLU stage for {file}: {e}")

    # 3. Save the integrated dataset
    # We export the structured information to the final JSON file    
    
    if final_results:
        print("\n--- STAGE 3: WRITE PROCESSED FILES ---")
        write_processed(final_results, os.path.join(PROCESSED_DIR, DEFAULT_INFORMATION_EXTRACTION_OUTPUT))
        print(f"Information extraction complete! Results saved to: {DEFAULT_INFORMATION_EXTRACTION_OUTPUT}")
        return DEFAULT_INFORMATION_EXTRACTION_OUTPUT
    else:
        print("\nPipeline finished but no data was extracted.")
        return None

def summary_generation(input_filename: str):
    print("\n--- STAGE 4: SUMMARY GENERATION ---")
    input_filename = input_filename or DEFAULT_INFORMATION_EXTRACTION_OUTPUT
    input_path = os.path.join(PROCESSED_DIR, input_filename + ".json")
    prompt_file_path = os.path.join(CONFIG_DIR, "summary_generation_prompt.txt")
    summarizer = SummaryGenerator(prompt_file_path)
    summarizer.generate_summaries_from_dataset(input_path, SUMMARY_DIR)
    print(f"\nSummary generation complete! Results saved to: {SUMMARY_DIR}")
    print(f"\n🚀 PIPELINE COMPLETE!")

if __name__ == "__main__":
    output_filename = None
    output_filename = information_extraction()
    summary_generation(output_filename)