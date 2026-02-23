import os
from utils.preprocessing import run_preprocessing, run_markdown_cleaning
from utils.extractor import Extractor
from utils.io import *
from utils.analysis import run_tf_id_analysis
import pandas as pd

DATA_DIR = "data"
CORPUS_DIR = "corpus"
PREPROCESSED_DIR = "preprocessed"
PROCESSED_DIR = "processed"
ADDITIONAL_ANALYSIS_DIR = "analysis"
SUMMARY_DIR = "summary"

def main():
    # 1. PDF to Markdown
    # Using marker-pdf to convert original PDFs into clean Markdown
    print("--- STAGE 1: PDF PREPROCESSING ---")
    # run_preprocessing("corpus")
    run_markdown_cleaning(f"{DATA_DIR}/{PREPROCESSED_DIR}")

    extractor = Extractor()
    # 1.5 TF-IDF Analysis (Optional)
    # This step is not strictly necessary for the final dataset, but we wanted to do it to see the insights it can provide.
    run_tf_id_analysis(f"{DATA_DIR}/{PREPROCESSED_DIR}", f"{DATA_DIR}/{ADDITIONAL_ANALYSIS_DIR}", extractor)

    # 2. Information Extraction Stage
    # This stage uses Hugging Face QA and SpaCy NER
    print("\n--- STAGE 2: NLU DATA EXTRACTION (HF + SpaCy) ---")
    export_dir = f"{DATA_DIR}/{PREPROCESSED_DIR}"
    final_results = []

    if not os.path.exists(export_dir):
        print(f"Error: {export_dir} folder not found. Preprocessing might have failed.")
        return

    # Navigate through the subfolders created by marker-pdf
    for folder in os.listdir(export_dir):
        folder_path = os.path.join(export_dir, folder)
        md_file_path = os.path.join(folder_path, "text.md")
        if not os.path.isdir(folder_path):
            continue
        if os.path.exists(md_file_path):       
            print(f"Extracting data from: {folder}...")
            try:
                extracted_json = extractor.extract_scholarship_data(md_file_path)
                
                if extracted_json:
                    final_results.append(extracted_json)
            except Exception as e:
                print(f"Error in NLU stage for {folder}: {e}")
        else:
            print(f"Warning: No 'text.md' found in {folder_path}")

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