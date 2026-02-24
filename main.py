import os
from utils.preprocessing import run_preprocessing, run_markdown_cleaning
from utils.extractor import Extractor
from utils.io import *
from utils.analysis import run_document_term_scoring

DATA_DIR = "data"
CORPUS_DIR = os.path.join(DATA_DIR,"corpus")
PREPROCESSED_DIR = os.path.join(DATA_DIR,"preprocessed")
PROCESSED_DIR = os.path.join(DATA_DIR,"processed")
ADDITIONAL_ANALYSIS_DIR = os.path.join(DATA_DIR,"analysis")
SUMMARY_DIR = os.path.join(DATA_DIR,"summary")

def main():
    # 1. PDF to Markdown
    # Using marker-pdf to convert original PDFs into clean Markdown
    print("--- STAGE 1: PDF PREPROCESSING ---")
    run_preprocessing(CORPUS_DIR, PREPROCESSED_DIR)
    run_markdown_cleaning(PREPROCESSED_DIR)

    extractor = Extractor()
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
        if file.endswith(".md"):
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
        output_filename = "final_scholarship_dataset"
        write_processed(final_results, os.path.join(PROCESSED_DIR , output_filename))
        
        print(f"\n🚀 PIPELINE COMPLETE!")
        print(f"Total documents processed: {len(final_results)}")
        print(f"Results saved to: {output_filename}")
    else:
        print("\nPipeline finished but no data was extracted.")

if __name__ == "__main__":
    main()