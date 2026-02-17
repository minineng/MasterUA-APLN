import os
import json
import ollama

def extract_scholarship_data(file_path):
    """
    Reads a Markdown file and uses Llama 3.2 to extract 
    structured information from BOE scholarship announcements.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    # System and user prompts in English
    # We limit content to 12,000 characters to stay within local LLM context limits
    user_prompt = f"""
    Analyze this BOE (Official State Gazette) document regarding student scholarships and extract technical information.
    Focus on the academic year, financial aid amounts, income thresholds, and deadlines.
    
    Return ONLY a JSON object with the following structure:
    {{
      "document_name": "filename",
      "academic_year": "e.g., 2024-2025",
      "aid_amounts": {{
        "income_linked": "fixed amount",
        "residence_linked": "fixed amount",
        "academic_excellence": "range or fixed amount"
      }},
      "income_thresholds": [
        {{ "family_members": 1, "threshold_1": 0, "threshold_2": 0, "threshold_3": 0 }},
        {{ "family_members": 2, "threshold_1": 0, "threshold_2": 0, "threshold_3": 0 }},
        {{ "family_members": 3, "threshold_1": 0, "threshold_2": 0, "threshold_3": 0 }}
      ],
      "application_deadline": "DD/MM/YYYY"
    }}

    Document content:
    {content[:12000]}
    """

    try:
        # Calling local Ollama instance
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {'role': 'system', 'content': 'You are a precise data extractor. You only output valid JSON.'},
                {'role': 'user', 'content': user_prompt}
            ],
            format='json',
            options={'temperature': 0.1} # Low temperature for higher factual accuracy
        )
        
        # Parse and return the JSON content
        return json.loads(response['message']['content'])
    
    except Exception as e:
        print(f"Error processing {file_path} with LLM: {e}")
        return None

def main():
    # 1. Path configuration (input folder from your teammate's branch)
    input_folder = 'export' 
    output_file = 'extracted_scholarships.json'
    
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' not found. Please run the preprocessing script first.")
        return

    # 2. List Markdown files
    md_files = [f for f in os.listdir(input_folder) if f.endswith('.md')]
    
    if not md_files:
        print("ℹ️ No .md files found in the input folder.")
        return

    print(f"🚀 Starting extraction of {len(md_files)} documents using Llama 3.2...")
    
    final_results = []

    # 3. Process each file
    for filename in md_files:
        file_path = os.path.join(input_folder, filename)
        print(f"--- Processing: {filename} ---")
        
        data = extract_scholarship_data(file_path)
        if data:
            # Add the filename to the data for traceability
            data['document_source'] = filename
            final_results.append(data)

    # 4. Save the final JSON results
    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(final_results, f_out, indent=4, ensure_ascii=False)
        print(f"\nExtraction complete. Structured data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving the output JSON: {e}")

if __name__ == "__main__":
    main()