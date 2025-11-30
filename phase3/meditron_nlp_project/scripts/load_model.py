from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


# Use Meditron-2 (ungated). Works in Colab Free without HF login.
MODEL_NAME = "epfl-llm/meditron-2-7b"


def load_meditron(device_map="auto"):
    # 4-bit quantization settings for Colab T4 free GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model (this may take ~20–40 seconds)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16
    )

    print("Model loaded on:", model.device)
    return model, tokenizer


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    
    model, tokenizer = load_meditron()
    print("Loaded model:", type(model))
