from math import nan
import unicodedata
from xmlrpc.client import MAXINT

import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm
import pathlib
import os
import re

def pdf_process(input_file, output_file):
    md_text = pymupdf4llm.to_markdown(input_file, footer=False, header=False, show_progress=True)
    #Extra cleanup
    # Normalize newlines
    md_text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove picture placeholders
    md_text = re.sub(r"\*\*==> picture .*? intentionally omitted <==\*\*", r"", md_text)
    # Remove page number
    md_text = re.sub(r"\n *?\d+ *?\n", r"\n", md_text)
    # Fix title
    md_text = re.sub(r"\n## *\*\*(.*?)\*\* *\n+## *\*\*(.*?)\*\* *\n", r"\n## **\1\2**\n", md_text)
    # Regenerate broken sentences
    matches = MAXINT
    while matches != 0:
        md_text, matches = re.subn(r"\n([^#\n][^\n]*?[\w,]) ?\n+ ?(\w)", r"\n\1 \2", md_text) 
    
    pathlib.Path(output_file).write_bytes(md_text.replace("\n", os.linesep).encode())
    print(f"Processed {input_file} -> {output_file}")


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
            filename = file.replace('.pdf', '')
            output_file = os.path.join(output_folder, f"{filename}.md")
            if os.path.exists(output_file):
                print(f'Skipping {input_file} as {output_file} already exists.')
                continue
            pdf_process(input_file, output_file)
            
    print("Preprocessing finished.")

REMOVE_LINE_PATTERNS = [
    r"^!\[.*\]\(.*\)\s*$",
    r"^Código seguro de Verificación",
    r"^CSV\s*:",
    r"^DIRECCIÓN DE VALIDACIÓN",
    r"^Puede verificar la integridad",
    r"^\s*\d+\s*$",
    r"^BOLET[IÍ]N OFICIAL DEL ESTADO",
    r"^Núm\.\s*\d+",
    r"^\*\*CSV.*"
]

# Define headers to remove entire sections
REMOVE_SECTION_HEADERS = [
    r"^## \*\*RESOLUCIÓN DE LA SECRETARÍA DE ESTADO DE EDUCACIÓN.*",
    r"^## Artículo 41.*",
    r"^## Artículo 42.*",
    r"^## Artículo 46.*",
    r"^## Artículo 66.*",
    r"^## Artículo 67.*",
    r"^## Artículo 69.*",
    r"^## Artículo 70.*",
    r"^## Artículo 71.*",
]

def clean_markdown_text(text: str) -> str:
    """
    Clean Markdown text:
    - Normalize line endings
    - Optionally keep title
    - Remove lines matching REMOVE_LINE_PATTERNS  
    - Remove entire sections matching header patterns
    - Dehyphenate words split across lines
    - Crop to relevant content
    """

    # Step 1: Remove entire sections based on headers
    lines = text.split("\n")
    keep_lines = []
    in_remove_section = False
    remove_header_level = 0  # Track level of section to remove (1=H1, 2=H2, etc.)

    for line in lines:
        s = line.strip()
        
        # Count # to determine header level
        header_level = len(s) - len(s.lstrip('#'))
        
        # Check if this is a header to remove
        is_remove_header = False
        for pat in REMOVE_SECTION_HEADERS:
            if header_level >= 1 and re.match(pat, s, re.IGNORECASE):
                is_remove_header = True
                remove_header_level = header_level  # Remember level to remove
                break
        
        if is_remove_header:
            in_remove_section = True  # Start removing
            continue
        
        if in_remove_section:
            # Stop at header of SAME LEVEL or HIGHER
            if header_level >= 1 and header_level <= remove_header_level:
                in_remove_section = False  # End removal
            continue
        
        keep_lines.append(line)

    text = "\n".join(keep_lines)


    # Step 2: Remove individual lines matching patterns
    cleaned_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue

        if any([re.search(pat, s) for pat in REMOVE_LINE_PATTERNS ]):
            continue

        s = re.sub(r"\s{2,}", " ", s)
        cleaned_lines.append(s)

    text2 = "\n".join(cleaned_lines)

    # Step 3: Dehyphenation: "gene-\nral" -> "general"
    text2 = re.sub(r"(\w)-\n(\w)", r"\1\2", text2)

    # Step 4: Cleanup text
    
    # 1. Remove table artifacts & markdown
    text2 = re.sub(r'(\|\ *?-+?\ *?\|)|(-+?\ *?\|)|([•|]+)|(—{2,})', ' ', text2)

    # 2. Remove HTML
    text2 = re.sub(r'<br>(.*?)<br>', r'\1', text2)
    text2 = re.sub(rf'(\. ){2,}', r' ', text2)

    # 3. Normalize whitespace first
    text2 = re.sub(r'\ +', ' ', text2)
    text2 = re.sub(r'\n+', '\n', text2)

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
        print(f"Cleaned: {input_md_path} -> {output_md_path}")


def run_markdown_cleaning(preprocessed_dir: str) -> None:
    if not os.path.exists(preprocessed_dir):
        print(f"Error: {preprocessed_dir} not found.")
        return

    for file in os.listdir(preprocessed_dir):
        if file.endswith(".md") and not file.endswith(".clean.md"):
            in_md = os.path.join(preprocessed_dir, file)
            filename = file.replace('.md', '')
            if not os.path.exists(in_md):
                continue

            out_md = os.path.join(preprocessed_dir, f"{filename}.clean.md")
            if os.path.exists(out_md):
                print(f'Skipping {in_md} as {out_md} already exists.')
                continue

            clean_markdown_file(in_md, out_md)
