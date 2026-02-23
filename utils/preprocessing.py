import os
import re
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

START_MARKERS = [
    r"\bCAP[IÍ]TULO\s+I\b",
    r"\bCap[ií]tulo\s+I\b",
    r"\bCAP[IÍ]TULO\s+PRIMERO\b",
    r"\bT[ÍI]TULO\s+I\b",
    r"\bArtículo\s+1\b",
    r"\bART[IÍ]CULO\s+1\b",
]

REMOVE_LINE_PATTERNS = [
    r"^!\[.*\]\(.*\)\s*$",
    r"^Código seguro de Verificación",
    r"^CSV\s*:",
    r"^DIRECCIÓN DE VALIDACIÓN",
    r"^Puede verificar la integridad",
    r"^\s*\d+\s*$",
    r"^BOLET[IÍ]N OFICIAL DEL ESTADO",
    r"^Núm\.\s*\d+",
]

def clean_markdown_text(text: str, keep_title: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Extract title before cleaning/cropping (optional)
    title_block = ""
    if keep_title:
        m = re.search(r"(?m)^(#\s+.*)$", text)
        if m:
            title_block = m.group(1).strip() + "\n\n"

    cleaned_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            cleaned_lines.append("")
            continue

        skip = False
        for pat in REMOVE_LINE_PATTERNS:
            if re.search(pat, s):
                skip = True
                break
        if skip:
            continue

        s = re.sub(r"\s{2,}", " ", s)
        cleaned_lines.append(s)

    text2 = "\n".join(cleaned_lines)

    # Dehyphenation: "gene-\nral" -> "general"
    text2 = re.sub(r"(\w)-\n(\w)", r"\1\2", text2)

    # Crop to relevant start
    start_idx = None
    for pat in START_MARKERS:
        m = re.search(pat, text2)
        if m:
            start_idx = m.start()
            break
    if start_idx is not None:
        text2 = text2[start_idx:]

    text2 = re.sub(r"\n{3,}", "\n\n", text2).strip() + "\n"

    if keep_title and title_block:
        return title_block + text2
    return text2

def clean_markdown_file(input_md_path: str, output_md_path: str) -> None:
    with open(input_md_path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = clean_markdown_text(raw)

    out_dir = os.path.dirname(output_md_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(cleaned)


def run_markdown_cleaning(preprocessed_dir: str) -> None:
    if not os.path.exists(preprocessed_dir):
        print(f"Error: {preprocessed_dir} not found.")
        return

    for folder in os.listdir(preprocessed_dir):
        folder_path = os.path.join(preprocessed_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        in_md = os.path.join(folder_path, "text.md")
        in_raw_md = os.path.join(folder_path, "text_raw.md")
        if not os.path.exists(in_md):
            continue

        os.rename(in_md, in_raw_md)
        out_md = os.path.join(folder_path, "text.md")

        clean_markdown_file(in_raw_md, out_md)
        print(f"Cleaned: {folder} -> {out_md}")
