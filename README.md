# Multilingual Intent Detection & Slot Filling

A comprehensive Natural Language Understanding (NLU) system that performs intent classification and slot filling across 51 languages using XLM-RoBERTa and CRF-based sequence labeling.

## 🌟 Features

- **Multilingual Support**: Handles 51 languages from the MASSIVE dataset
- **Intent Classification**: 60 intent classes using fine-tuned XLM-RoBERTa
- **Slot Filling**: 55 slot types with CRF-enhanced sequence labeling
- **Interactive Web Interface**: Streamlit-based demo application
- **Zero-shot & Few-shot Evaluation**: Cross-lingual performance analysis

## 🏗️ Architecture

### Intent Classification
- **Model**: XLM-RoBERTa-base fine-tuned for sequence classification
- **Task**: Multi-class intent classification (60 classes)
- **Training**: Fine-tuned on MASSIVE dataset with multilingual examples

### Slot Filling
- **Model**: XLM-RoBERTa-base with CRF layer
- **Tagging Scheme**: BIO (Beginning, Inside, Outside)
- **Task**: Named Entity Recognition / Slot extraction (55 slot types)
- **Enhancement**: Conditional Random Field (CRF) for sequence labeling

## 📋 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Running the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Models

The application loads pre-trained models from:
- Intent Classifier: `xlm-roberta-intent-classifier-final/`
- Slot Filling Model: `slot_filling_model_crf/final_model/`

## 📊 Model Performance

### Intent Classification
- Overall Accuracy: See evaluation results in `xlm-roberta-intent-classifier-final/evaluation_summary.json`
- Per-language accuracy available in `per_language_accuracy.csv`
- Per-intent F1 scores available in `per_class_f1_scores.csv`

### Slot Filling
- Overall F1 Score: See `slot_filling_model_plaintext/test_results.json`
- Supports 55 slot types with BIO tagging

## 📁 Project Structure

```
intent_project/
├── app.py                          # Streamlit web application
├── generate_visualizations.py      # Evaluation visualization scripts
├── generate_slot_filling_flowchart.py  # Flowchart generation
├── NLP_IntentClassification.ipynb  # Intent model training notebook
├── NLP_SlotFilling_1st.ipynb      # Slot filling model training notebook
├── hyperparameters_report.md       # Training hyperparameters documentation
├── hyperparameters_table.md       # Hyperparameters in table format
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔬 Training

Training notebooks are provided for both models:
- `NLP_IntentClassification.ipynb`: Intent classification model training
- `NLP_SlotFilling_1st.ipynb`: Slot filling model training

See `hyperparameters_report.md` for detailed training configurations.

## 📈 Evaluation

Evaluation results and visualizations can be generated using:
```bash
python generate_visualizations.py --model-dir xlm-roberta-intent-classifier-final/
```

## 🎯 Supported Languages

The system supports 51 languages including:
- English, Spanish, French, German, Italian, Portuguese
- Chinese (Simplified & Traditional), Japanese, Korean
- Arabic, Hindi, Bengali, Tamil, Telugu, and many more

See the `data/` directory for the complete list of supported language files.

## 📝 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- **MASSIVE Dataset**: Multilingual dataset for NLU tasks
- **XLM-RoBERTa**: Base transformer model from Hugging Face
- **Hugging Face Transformers**: Model training and inference framework

## 📧 Contact

For questions or issues, please open an issue in the repository.

