from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "epfl-llm/meditron-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={
        "cuda:0": "6GB",
        "cpu": "20GB"
    }
)

print("Loaded Meditron successfully.")
