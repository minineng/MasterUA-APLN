import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm
import pathlib
import os
import re

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
            if os.path.exists(output_file):
                print(f'Skipping {input_file} as {output_file} already exists.')
                continue
            pdf_process(input_file, output_file)
            
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
