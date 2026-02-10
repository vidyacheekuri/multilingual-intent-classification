# Multilingual Intent Detection & Slot Filling

A **Natural Language Understanding (NLU)** system that performs intent classification and slot filling in **51 languages**, built with XLM-RoBERTa and a CRF-based sequence labeler. Includes a Streamlit app for real-time inference.

---

## What this project does

Voice assistants and chatbots need to turn a user’s sentence into something actionable: *what* the user wants (intent) and *which pieces of information* are in the sentence (slots). This repo implements that pipeline in a single, multilingual setup.

**Example:**  
For *"Play jazz music by Miles Davis tomorrow at 6 PM"* the system predicts:

- **Intent:** `play_music`  
- **Slots:** `MUSIC_GENRE` = "jazz", `ARTIST_NAME` = "Miles Davis", `DATE_TIME` = "tomorrow at 6 PM"

So you get both a label for the user goal and a structured set of (slot type, value) pairs that downstream code can use (e.g. to call a music API or set a reminder).

---

## How it works

- **Intent classification**  
  A single label is predicted for the whole utterance (e.g. `alarm_set`, `weather_query`, `play_music`). We use a **fine-tuned XLM-RoBERTa-base** model with a classification head, trained to choose among **60 intents** from the MASSIVE schema.

- **Slot filling**  
  Each token (or word) is tagged with a label in **BIO** form: B-*SlotType* (beginning of a span), I-*SlotType* (inside), or O (outside any slot). The model is again XLM-RoBERTa-base, with a **CRF layer** on top to enforce valid transitions (e.g. I-ARTIST only after B-ARTIST or I-ARTIST). There are **55 slot types** (e.g. `ARTIST_NAME`, `TIME`, `LOCATION`).

- **Multilingual**  
  Both models are trained on **MASSIVE** (Amazon Science), so the same weights handle **51 languages** without separate models per language.

The Streamlit app runs both models in sequence: you type or paste an utterance and see the predicted intent (with confidence) and the extracted slots.

---

## Dataset and training

Training data comes from the **MASSIVE** dataset: parallel, multilingual intent-and-slot annotations for the same set of intents and slot types across many languages. Intent model: 5 epochs, F1-macro early stopping. Slot model: 3 epochs with the CRF. Details and hyperparameters are in the training notebooks (`NLP_IntentClassification.ipynb`, `NLP_SlotFilling_1st.ipynb`).

---

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
