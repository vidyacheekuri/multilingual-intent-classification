# Multilingual Intent Detection & Slot Filling

NLU system for intent classification and slot filling across **51 languages** using XLM-RoBERTa and CRF-based sequence labeling (MASSIVE dataset). Includes a Streamlit app for real-time inference.

## Live Demo

**[Try the app →](https://multilingual-intent-classification.streamlit.app)**

## Features

- **51 languages**, **60 intent classes**, **55 slot types** (BIO tagging)
- Intent classification with confidence scores
- CRF-enhanced slot extraction
- Streamlit web UI; models load from Hugging Face Hub on Streamlit Cloud

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Models run from local folders if present, or from Hugging Face when `INTENT_MODEL_HF_REPO` / `SLOT_MODEL_HF_REPO` (or Streamlit secrets) are set.

## Repo contents

- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies
- `NLP_IntentClassification.ipynb` — Intent model training
- `NLP_SlotFilling_1st.ipynb` — Slot filling (CRF) training
- Other notebooks — Evaluation and experiments

## License

For research and educational use.

## Acknowledgments

MASSIVE dataset (Amazon Science), XLM-RoBERTa (Hugging Face), Streamlit, PyTorch.
