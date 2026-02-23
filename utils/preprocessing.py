import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm
import pathlib
import os

def pdf_process(input_file, output_file):
    print(f"Processing {input_file}")
    md_text = pymupdf4llm.to_markdown(input_file)
    pathlib.Path(output_file).write_bytes(md_text.encode())


def run_preprocessing(input_folder="corpus", output_folder="preprocessed"):
    print("Starting PDF preprocessing...")
    
    # Check if input directory exists
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' not found.")
        return

    # Process all PDFs in the corpus folder
    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            print(f"Processing {file}")
            input_file = os.path.join(input_folder, file)
            output_file = f"{output_folder}/{file.replace('.pdf', '.md')}"
            pdf_process(input_file, output_file)
            
    print("Preprocessing finished.")

if __name__ == "__main__":
    folder_path = "./corpus"
    output_folder = "./processed"
    run_preprocessing(folder_path, output_folder)
    