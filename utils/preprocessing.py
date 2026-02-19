import os
import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import pypdfium2 as pdfium

from utils.io import *

def pdf_process(ruta_pdf, carpeta_salida="resultado_practica"):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    resultados = {
        "text_path": os.path.join(carpeta_salida, "contenido_practica.txt"),
        "images": []
    }

    pdf = pdfium.PdfDocument(ruta_pdf)
    
    # Abrimos el archivo de texto
    with open(resultados["text_path"], "w", encoding="utf-8") as f_txt:
        
        for i, pagina in enumerate(pdf):
            num_pag = i + 1
            f_txt.write(f"--- PÁGINA {num_pag} ---\n")
            
            # Text Extraction
            textpage = pagina.get_textpage()
            f_txt.write(textpage.get_text_range() + "\n\n")

            # Image Extraction
            for j, obj in enumerate(pagina.get_objects()):
                if isinstance(obj, pdfium.PdfImage):
                    try:
                        ruta_img = os.path.join(carpeta_salida, f"img_p{num_pag}_obj{j}.png")
                        bitmap = obj.get_bitmap()
                        if bitmap:
                            imagen_pil = bitmap.to_pil()
                            imagen_pil.save(ruta_img)
                            resultados["images"].append(ruta_img)
                    except Exception as e:
                        print(f"No se pudo extraer la imagen {j} en pág {num_pag}: {e}")
            pagina.close()

    pdf.close()
    clean_duplicated_images(carpeta_salida)
    return resultados



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

if __name__ == "__main__":
    folder_path = "./corpus"
    output_folder = "./processed"
    for file_name in os.listdir(folder_path):
        file_path = f"{folder_path}/{file_name}"
        output_path = f"{output_folder}/{file_name.replace('.pdf', '')}"
        if not os.path.exists(output_path):
            print("Processing: ", file_name)
            pdf_data = pdf_process(file_path, output_path)
        else:
            print("Output folder already exists for this PDF, skipping: ", file_name)
    