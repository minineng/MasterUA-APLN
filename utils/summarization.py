import os
import json
import torch
import time
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.io import *

class SummaryGenerator:

    DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct" 
    DEFAULT_TOKENS_TO_GENERATE = 600

    def __init__(self, prompt_file_path: str, model_name: str = DEFAULT_MODEL):
        self.prompt_file_path = prompt_file_path
        self.model_name = model_name
        self.llm = self.create_llm()        

    def create_llm(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )

        model.to(device)
        model.eval()

        return {"model": model, "tokenizer": tokenizer, "device": device}
    
    def build_prompt_from_json(self, record: Dict[str, Any]) -> str:
        json_text = json.dumps(record, ensure_ascii=False)
        instruction = read_txt(self.prompt_file_path).format(json_text=json_text)

        return instruction

    def format_chat_input(self, tokenizer, prompt: str, max_input_tokens: int = 2048):
        if getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "Eres un asistente que genera resúmenes en español."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens
            )
            return enc.input_ids, enc.attention_mask

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        )
        return enc.input_ids, enc.attention_mask


    @torch.no_grad()
    def generate_summary(self,record: Dict[str, Any]) -> str:
        model = self.llm["model"]
        tokenizer = self.llm["tokenizer"]
        device = self.llm["device"]

        prompt = self.build_prompt_from_json(record)
        input_ids, attention_mask = self.format_chat_input(tokenizer, prompt, max_input_tokens=2048)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.DEFAULT_TOKENS_TO_GENERATE,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.08,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        gen = output_ids[0][input_ids.shape[-1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()

        return text


    def generate_summaries_from_dataset(self, input_json_path: str, out_dir: str) -> None:
        dataset: List[Dict[str, Any]] = read_json(input_json_path)

        for i, record in enumerate(dataset):
            starting_time = time.time()
            doc_name = str(record.get("nombre_documento", "")).strip()
            base = os.path.splitext(os.path.basename(doc_name))[0] if doc_name else f"doc_{i:02d}"

            summary = self.generate_summary(record)

            out_path = os.path.join(out_dir, f"{base}.summary.txt")
            write_txt(summary, out_path)
            print(f"OK - Summary for {base} completed in {time.time() - starting_time:.2f} seconds.")