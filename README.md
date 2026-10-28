# SupportTune: An Evaluated Customer-Support LLM Fine-Tuned with QLoRA

SupportTune is an experimental customer-support chatbot built by fine-tuning
Qwen2.5-0.5B-Instruct using 4-bit Quantized Low-Rank Adaptation (QLoRA).

The project covers dataset preparation, model fine-tuning, evaluation against
the base model, and a Streamlit interface for interactive testing.

## Features

- Customer-support question answering
- 4-bit NF4 quantization for memory-efficient fine-tuning
- LoRA adapter training with PEFT
- Train, validation, and test data separation
- Base-versus-fine-tuned response comparison
- Interactive Streamlit chatbot
- Evaluation outputs saved in CSV and JSON formats

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT and QLoRA
- bitsandbytes
- Google Colab with NVIDIA T4 GPU
- Streamlit

## Model and Training Configuration

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Dataset | Bitext customer-support chatbot dataset |
| Training examples used | 4,000 |
| Validation examples used | 400 |
| Training epochs | 1 |
| Training steps | 250 |
| Maximum sequence length | 512 tokens |
| Quantization | 4-bit NF4, double quantization |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Trainable parameters | 8,798,208 |
| Total parameters | 502,830,976 |
| Trainable parameter percentage | Approximately 1.75% |
| Final training loss | 0.8111 |
| Final validation loss | 0.7965 |

These results describe the recorded training run, not a guarantee of
production-level response quality.

## Dataset Preparation

The dataset contains customer-support instructions, categories, intents,
and reference responses. Examples were separated into training, validation,
and test sets, with identical instruction text grouped to reduce exact
instruction overlap between splits.

The saved splits contain approximately:

- Training: 21,482 examples
- Validation: 2,663 examples
- Test: 2,727 examples

The fine-tuning run used a subset of 4,000 training examples and 400
validation examples to fit the available Colab GPU resources.

## Evaluation

A set of 50 held-out test questions was generated for both the base and
fine-tuned models. Responses and reference answers were saved for comparison.

The evaluation artifact includes:
- Customer-support question and reference response
- Base-model answer
- Fine-tuned-model answer
- Category and intent

The evaluation supports qualitative comparison, but a comprehensive,
validated aggregate quality score has not yet been established. Lower
validation loss alone does not prove better customer-support answers.

Known limitations include occasional unsupported assumptions, incorrect
interpretations of ambiguous questions, and responses that may imply access
to systems the chatbot does not actually have.

## Running the Demo

The Streamlit interface is in `app.py`. The app loads the base model and
the trained LoRA adapter, then generates responses to user questions.

The current implementation uses Google Drive paths from the Colab
environment. Update `ADAPTER_PATH` and configure the model environment
before running it elsewhere.

Typical launch command in the configured environment:

```bash
streamlit run app.py
```

Required Python packages include:

```text
streamlit
torch
transformers
peft
bitsandbytes
accelerate
```

Compatible package versions and a supported GPU environment may be
required. The app has been tested in Google Colab; it is not yet a
permanently hosted public service.

## Safety and Limitations

- This is a portfolio and experimentation project, not a production
  customer-support system.
- It cannot access real orders, accounts, payment systems, or refund tools.
- Do not enter personal, financial, or confidential information.
- The model can produce inaccurate or unsupported answers.
- No claim of production readiness or statistically proven quality
  improvement is made.

## Project Artifacts

Training and evaluation artifacts are saved in the project's `results/`
directory, including:

- `qlora_r16_metrics.json`
- `qlora_r16_training_log.csv`
- `qlora_r16_config.json`
- `evaluation_50_examples.csv`
- `evaluation_50_examples.json`
- `base_vs_finetuned_comparison.csv`
- `base_vs_finetuned_comparison.json`

The trained adapter is stored separately under `models/`.

## Future Improvements

- Add automated evaluation metrics and human review
- Measure response relevance, correctness, and unsupported claims
- Improve handling of ambiguous requests
- Add retrieval-augmented generation (RAG) for verified support policies
- Add tests, configuration files, and reproducible setup instructions
- Deploy to a persistent hosting environment

## Author

Developed as a hands-on project in parameter-efficient LLM fine-tuning,
model evaluation, and chatbot development.
