import pandas as pd
import torch
from scripts.load_model import load_meditron


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ask_question(pipe_or_model_tokenizer, tokenizer, model, question, max_new_tokens=80):
# If using raw model + tokenizer
inputs = tokenizer(question, return_tensors="pt").to(DEVICE)
with torch.no_grad():
out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
resp = tokenizer.decode(out[0], skip_special_tokens=True)
return resp


if __name__ == "__main__":
df = pd.read_csv('data/medical_questions.csv')
model, tokenizer = load_meditron()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


preds = []
for q in df['question']:
r = ask_question(None, tokenizer, model, q)
preds.append(r)


df['response'] = preds
# Simple exact-match scoring (student-level). You can expand to fuzzy matching later.
def score_row(resp, gold):
return int(gold.lower() in resp.lower())


df['correct'] = df.apply(lambda r: score_row(r['response'], r['answer']), axis=1)
acc = df['correct'].mean()
print(f"Accuracy (simple substring method): {acc:.3f}")
df.to_csv('results/outputs/mcq_responses.csv', index=False)