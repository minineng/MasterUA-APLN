import os
from utils.preprocessing import run_preprocessing
from utils.extractor import Extractor
from utils.io import *

DATA_DIR = "data"
CORPUS_DIR = "corpus"
PREPROCESSED_DIR = "preprocessed"
PROCESSED_DIR = "processed"
SUMMARY_DIR = "summary"

def main():
    # 1. PDF to Markdown
    # Using marker-pdf to convert original PDFs into clean Markdown
    print("--- STAGE 1: PDF PREPROCESSING ---")
    run_preprocessing(os.path.join(DATA_DIR,CORPUS_DIR), os.path.join(DATA_DIR, PREPROCESSED_DIR))

    # 2. Information Extraction Stage
    # This stage uses Hugging Face QA and SpaCy NER
    print("\n--- STAGE 2: NLU DATA EXTRACTION (HF + SpaCy) ---")
    extractor = Extractor()
    export_dir = f"{DATA_DIR}/{PREPROCESSED_DIR}"
    final_results = []

    if not os.path.exists(export_dir):
        print(f"Error: {export_dir} folder not found. Preprocessing might have failed.")
        return

    # Navigate through the markdown created by pymupdf4llm
    for file in os.listdir(export_dir):
        if file.endswith(".md"):
            try:
                extracted_json = extractor.extract_scholarship_data(os.path.join(export_dir,file))
                if extracted_json:
                    final_results.append(extracted_json)
            except Exception as e:
                print(f"Error in NLU stage for {file}: {e}")

    # 3. Save the integrated dataset
    # We export the structured information to the final JSON file    
    
    if final_results:
        print("\n--- STAGE 3: WRITE PROCESSED FILES ---")
        output_filename = "final_scholarship_dataset"
        write_processed(final_results, f"{DATA_DIR}/{PROCESSED_DIR}/{output_filename}")
        
        print(f"\n🚀 PIPELINE COMPLETE!")
        print(f"Total documents processed: {len(final_results)}")
        print(f"Results saved to: {output_filename}")
    else:
        print("\nPipeline finished but no data was extracted.")

if __name__ == "__main__":
    main()