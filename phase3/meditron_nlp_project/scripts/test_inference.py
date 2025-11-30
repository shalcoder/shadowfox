import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/BioGPT"

print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

prompt = "COVID-19 is caused by"
print(f"Generating for prompt: '{prompt}'")

inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)

generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
print("-" * 30)
print(generated_text)
print("-" * 30)
print("Inference test complete.")
