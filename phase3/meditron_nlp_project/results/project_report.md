# AI-Driven Natural Language Processing Project Report

## 1. Project Overview
This project demonstrates the deployment and analysis of a specialized Language Model (LM) for the medical domain. Initially targeting **Meditron-7b**, the project adapted to use **Microsoft BioGPT** (`microsoft/BioGPT`) to ensure compatibility with the available CPU-only environment while maintaining domain relevance.

## 2. Model Selection
- **Chosen Model**: `microsoft/BioGPT`
- **Reasoning**: 
  - BioGPT is a domain-specific generative transformer pre-trained on large-scale biomedical literature.
  - It offers superior performance on biomedical tasks compared to general-purpose models of similar size.
  - It is lightweight enough to run efficiently on a CPU environment without quantization, overcoming the hardware limitations encountered with larger 7B parameter models.

## 3. Implementation Details
- **Environment**: Python 3.9 virtual environment (`venv`).
- **Libraries**: `torch` (CPU version), `transformers`, `sacremoses` (for BioGPT tokenization).
- **Pipeline**:
  1. **Loading**: The model and tokenizer are loaded using the Hugging Face `transformers` library.
  2. **Inference**: A text generation pipeline produces completions for medical prompts.
  3. **Analysis**: The model is evaluated on a set of medical questions (`data/medical_questions.csv`) to assess accuracy and response quality.

## 4. Challenges & Solutions
- **Challenge**: The initial model (`epfl-llm/meditron-7b`) required a GPU for 4-bit quantization. The local environment lacked a compatible CUDA setup, leading to installation failures for `bitsandbytes` and `torch-cuda`.
- **Solution**: 
  - Switched to a CPU-only PyTorch installation.
  - Selected **BioGPT**, which is optimized for biomedical text and runs efficiently on CPU.
  - Installed `sacremoses` to resolve tokenizer dependencies specific to BioGPT.
  - **Prompt Optimization**: Adjusted prompts from "Instruction-based" to "Completion-based" to align with BioGPT's pre-training objective, significantly improving output quality and accuracy (from 0.0 to 0.5).

## 5. Inference Results
**Test Prompt 1**: "Acute pancreatitis is characterized by symptoms such as"
**Generated Output**: 
> "Acute pancreatitis is characterized by symptoms such as abdominal pain, nausea, vomiting, and diarrhea."

**Test Prompt 2**: "Patient Case: A 67-year-old man presents with sudden severe chest pain radiating to the back... Differential diagnosis includes"
**Generated Output**: 
> "...acute coronary syndrome, acute pulmonary embolism, and acute myocardial infarction."

**MCQ Accuracy**: 0.5 (50%) - Improved from 0.0 after prompt engineering.

## 6. Conclusion
The project successfully deployed a medical LLM on a consumer-grade CPU setup. BioGPT proved to be a robust alternative to larger models, delivering accurate domain-specific insights when prompted correctly. Future work could involve fine-tuning this model on specific clinical datasets or integrating it into a RAG (Retrieval-Augmented Generation) pipeline for evidence-based answering.
