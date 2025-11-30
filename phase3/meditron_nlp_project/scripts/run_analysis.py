import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'outputs')

os.makedirs(RESULTS_DIR, exist_ok=True)

print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())

MODEL_NAME = "microsoft/BioGPT"

# CPU-friendly configuration (no quantization, float32)
print("Loading model on CPU...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32, # Use float32 for CPU
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    print('Model loaded')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def generate(prompt, max_new_tokens=80, temperature=0.3):
    inputs = tokenizer(prompt, return_tensors='pt').to('cpu')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("Testing generation...")
# BioGPT works best with sentence completions
print(generate("Acute pancreatitis is characterized by symptoms such as"))

prompt = '''Patient Case: A 67-year-old man presents with sudden severe chest pain radiating to the back. BP 90/60, sweating, tearing pain.
Differential diagnosis includes'''
print(generate(prompt, max_new_tokens=120))

try:
    long_text_path = os.path.join(DATA_DIR, 'sample_clinical_notes.txt')
    long_text = open(long_text_path).read()
    print(generate('Summarize the following clinical note in 3 bullets:\n' + long_text, max_new_tokens=120))
except FileNotFoundError:
    print(f"Warning: {long_text_path} not found")

print("Running MCQ evaluation...")
try:
    mcq_path = os.path.join(DATA_DIR, 'medical_questions.csv')
    mcq = pd.read_csv(mcq_path)
    # Format as "Question: [q] Answer:" to guide the model
    mcq['response'] = mcq['question'].apply(lambda q: generate(f"Question: {q}\nAnswer:", max_new_tokens=20))
    mcq['len'] = mcq['response'].str.len()
    mcq.to_csv(os.path.join(RESULTS_DIR, 'mcq_responses.csv'), index=False)
    
    mcq['correct'] = mcq.apply(lambda r: int(r['answer'].lower() in r['response'].lower()), axis=1)
    print('Accuracy:', mcq['correct'].mean())
    
    mcq.to_csv(os.path.join(RESULTS_DIR, 'mcq_responses_with_scores.csv'), index=False)
    print(f'Saved outputs to {RESULTS_DIR}')
except Exception as e:
    print(f"Error in MCQ evaluation: {e}")

print("Running LangChain demo...")
try:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.15
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    template = """Question: {question}

Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    chain = prompt | llm

    question = "What is the mechanism of action of Aspirin?"
    print(chain.invoke({"question": question}))
except Exception as e:
    print(f"Error in LangChain demo: {e}")

print("Analysis complete.")
