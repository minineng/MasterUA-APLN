import json
import requests
import os

class SummaryGenerator:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
        self.model = model_name
        self.api_url = f"{base_url}/api/generate"

    def process_dataset(self, dataset_path, output_dir):
        # Reads the final_scholarship_dataset.json and generates individual summaries for each academic year found

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            print(f"Dataset loaded. Processing {len(dataset)} entries...")

            for entry in dataset:
                year_label = entry.get("año_academico", "unknown").replace("/", "-")
                print(f"Generating Spanish summary for: {year_label}...")
                
                # Execution of the generation logic
                summary_text = self._request_llm_summary(entry)
                
                # Saving the result
                filename = f"summary_{year_label}.txt"
                save_path = os.path.join(output_dir, filename)
                with open(save_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(summary_text)
                
                print(f"Successfully saved to: {save_path}")

        except Exception as e:
            print(f"Error during dataset processing: {e}")

    def _request_llm_summary(self, data):
        json_context = json.dumps(data, indent=2, ensure_ascii=False)
        
        # SPANISH PROMPT (to ensure Ollama understands and responds in Spanish)
        spanish_prompt = f"""
        Eres un experto en el Boletín Oficial del Estado (BOE) y becas del Ministerio de Educación.
        A continuación tienes los datos extraídos de la convocatoria {data.get('año_academico')}:
        
        DATOS EXTRAÍDOS (JSON):
        {json_context}
        
        TU TAREA:
        Redacta un resumen profesional y claro teniendo en cuenta los datos extraídos del JSON.
        
        INSTRUCCIONES CRÍTICAS:
        1. Si un campo contiene ruido legal como "Artículo 1" o "DIRECCIÓN DE VALIDACIÓN", ignóralo completamente.
        2. Describe las cuantías fijas (renta, residencia y básica) y menciona los premios por excelencia académica.
        3. Si la cuantía de residencia ha cambiado (ej. de 1600€ a 2500€ o 2700€), menciónalo.
        4. Indica los plazos de solicitud (inicio y fin) de forma clara.
        5. No inventes información; si un dato es erróneo en el JSON, utiliza una descripción general.

        RESUMEN EN ESPAÑOL:
        """

        payload = {
            "model": self.model,
            "prompt": spanish_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2  # High precision
            }
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "Error: Couldn't generate the summary.")
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

if __name__ == "__main__":
    # Path configuration
    INPUT_FILE = "data/processed/final_scholarship_dataset.json"
    OUTPUT_DIR = "data/processed/summaries"

    # Initialization
    generator = SummaryGenerator(model_name="llama3")
    generator.process_consolidated_dataset(INPUT_FILE, OUTPUT_DIR)