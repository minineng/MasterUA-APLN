import os
import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

def export_pdf(file, content):
    # Create folder for this PDF (remove .pdf extension from folder name)
    folder_name = file.replace(".pdf", "")
    export_path = os.path.join("export", folder_name)
    os.makedirs(export_path, exist_ok=True)
    
    # Save text content
    with open(os.path.join(export_path, "text.md"), "w", encoding="utf-8") as f:
        f.write(content[0])
    
    # Save images (maybe we could delete this part?? there are only QR and logos)
    for image_name, image_content in content[2].items():
        image_content.save(os.path.join(export_path, image_name))

    print(f"Exported {file} contents to {export_path}")

def run_preprocessing(input_folder="corpus"):
    print("Starting PDF preprocessing...")
    torch.cuda.empty_cache()
    
    # Force CPU
    device = "cpu"

    # Initialize converter with model dictionary
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={
            "workers": 1, 
            "device": device
        }
    )

    # Check if input directory exists
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' not found.")
        return

    # Process all PDFs in the corpus folder
    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            print(f"Processing {file}")
            file_path = os.path.join(input_folder, file)
            
            # Execute conversion logic
            rendered = converter(file_path)
            content = text_from_rendered(rendered)
            
            # Call the export function
            export_pdf(file, content)
            
            # Memory management
            torch.cuda.empty_cache()
            
    print("Preprocessing finished.")