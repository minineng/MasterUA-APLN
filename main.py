import os
import json
from preprocessing import run_preprocessing
from extractor import extract_scholarship_data

def main():
    # 1. PDF to Markdown
    # This will create 'export/document_name/text.md' for each PDF
    print("--- STAGE 1: PDF PREPROCESSING ---")
    # run_preprocessing("corpus")

    # 2. Information Extraction Stage (Your NLU logic)
    print("\n--- STAGE 2: NLU DATA EXTRACTION ---")
    export_dir = "export"
    final_results = []

    if not os.path.exists(export_dir):
        print(f"Error: {export_dir} folder not found. Preprocessing might have failed.")
        return

    # Navigate through the subfolders created by marker-pdf
    for folder in os.listdir(export_dir):
        folder_path = os.path.join(export_dir, folder)
        md_file_path = os.path.join(folder_path, "text.md")
        
        if os.path.exists(md_file_path):
            print(f"Processing text with Llama 3.2: {folder}...")
            # We call your extraction function
            extracted_json = extract_scholarship_data(md_file_path)
            
            if extracted_json:
                final_results.append(extracted_json)
        else:
            print(f"Warning: No 'text.md' found in {folder_path}")

    # 3. Save the integrated dataset
    output_filename = "final_scholarship_dataset.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n PIPELINE COMPLETE!")
    print(f"Total documents processed: {len(final_results)}")
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    main()