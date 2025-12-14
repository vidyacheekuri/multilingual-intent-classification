# Multilingual Intent Detection & Slot Filling

A comprehensive Natural Language Understanding (NLU) system that performs intent classification and slot filling across 51 languages using XLM-RoBERTa and CRF-based sequence labeling. This project includes both the trained models and an interactive web application for real-time inference.

## Project Overview

This project implements a complete multilingual NLU pipeline that can:
- Classify user intents across 60 different intent classes
- Extract slot values using BIO tagging scheme across 55 slot types
- Process utterances in 51 different languages
- Provide real-time inference through a Streamlit web interface

The system is built on the MASSIVE dataset and uses state-of-the-art transformer architectures with CRF enhancement for sequence labeling tasks.

## Features

### Core Capabilities
- **Multilingual Support**: Handles 51 languages from the MASSIVE dataset including English, Spanish, French, German, Italian, Portuguese, Chinese (Simplified & Traditional), Japanese, Korean, Arabic, Hindi, Bengali, Tamil, Telugu, and many more
- **Intent Classification**: 60 intent classes using fine-tuned XLM-RoBERTa-base model
- **Slot Filling**: 55 slot types with CRF-enhanced sequence labeling using BIO tagging scheme
- **Interactive Web Interface**: Streamlit-based demo application with modern UI
- **Real-time Inference**: Fast prediction with GPU support and CPU fallback
- **Zero-shot & Few-shot Evaluation**: Cross-lingual performance analysis capabilities

### Web Application Features
- Clean, modern interface with gradient backgrounds and card-based layouts
- Real-time intent detection with confidence scores
- Slot extraction with visual tag display
- Example utterance dropdown for quick testing
- Sidebar with key metrics (51 languages, 60 intents, 55 slot types)
- Responsive design optimized for single-screen viewing
- Support for multiple languages in a single interface

## Architecture

### Intent Classification Model

**Base Architecture**: XLM-RoBERTa-base
- Model Size: 768 hidden dimensions, 12 transformer layers, 12 attention heads
- Vocabulary Size: 250,002 tokens
- Number of Classes: 60 intents
- Max Sequence Length: 128 tokens

**Training Details**:
- Learning Rate: 2e-5 (0.00002)
- Optimizer: AdamW
- Train Batch Size: 16 per device
- Evaluation Batch Size: 32 per device
- Number of Epochs: 5
- Weight Decay: 0.01 (L2 regularization)
- Warmup Steps: 500
- Mixed Precision Training: FP16 enabled
- Random Seed: 42
- Best Model Metric: F1-macro
- Training Samples: 230,280
- Validation Samples: 40,660
- Test Samples: 59,480

**Model Location**: `xlm-roberta-intent-classifier-final/`

### Slot Filling Model

**Base Architecture**: XLM-RoBERTa-base with CRF layer
- Model Size: 768 hidden dimensions, 12 transformer layers, 12 attention heads
- Vocabulary Size: 250,002 tokens
- Number of Labels: 111 (BIO tags: 55 slot types × 2 + 1 for 'O')
- Slot Types: 55 unique slot types
- Max Sequence Length: 128 tokens

**Tagging Scheme**: BIO (Beginning, Inside, Outside)
- B- prefix: Beginning of a slot
- I- prefix: Inside/continuation of a slot
- O: Outside/not a slot

**Training Details**:
- Learning Rate: 2e-5 (0.00002)
- Optimizer: AdamW
- Train Batch Size: 32 per device
- Evaluation Batch Size: 64 per device
- Number of Epochs: 3
- Weight Decay: 0.01 (L2 regularization)
- Warmup Steps: 200
- Mixed Precision Training: FP16 enabled
- Random Seed: 42
- Best Model Metric: F1 score
- CRF Layer: Conditional Random Field for sequence labeling to enforce valid BIO tag transitions

**Model Location**: `slot_filling_model_crf/final_model/`

### CRF Implementation

The slot filling model uses a custom SimpleCRF class implemented in PyTorch that:
- Enforces valid BIO tag transitions (e.g., I-tag must follow B-tag or I-tag of same type)
- Uses Viterbi decoding for optimal sequence prediction
- Handles variable-length sequences with masking
- Computes transition scores between consecutive tags

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- 8GB+ RAM recommended
- 5GB+ disk space for models

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd intent_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
   - The models should be placed in:
     - Intent Classifier: `xlm-roberta-intent-classifier-final/`
     - Slot Filling Model: `slot_filling_model_crf/final_model/`
   - Note: Model files are large and should be downloaded separately or stored in cloud storage

## Usage

### Running the Web Application Locally

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Models Programmatically

The application loads models on startup. Key functions:

- `predict_intent(utterance)`: Returns (intent, confidence) tuple
- `extract_slots(utterance)`: Returns list of (slot_type, slot_value) tuples

Example:
```python
from app import predict_intent, extract_slots

intent, confidence = predict_intent("Play jazz music by Miles Davis")
slots = extract_slots("Play jazz music by Miles Davis")
```

## Deployment

### Deploying to Railway

Railway is a modern platform for deploying applications. Here's how to deploy this Streamlit app:

1. **Create a Railway Account**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Create a New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure the Deployment**
   - Railway will auto-detect it's a Python project
   - Add the following environment variables if needed:
     - `PYTHON_VERSION=3.10`
     - `PORT=8501` (Streamlit default)

4. **Create railway.json (Optional)**
   Create a `railway.json` file in the root:
   ```json
   {
     "$schema": "https://railway.app/railway.schema.json",
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "streamlit run app.py --server.port $PORT --server.address 0.0.0.0",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 10
     }
   }
   ```

5. **Create Procfile (Alternative)**
   Create a `Procfile` in the root:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

6. **Model Storage**
   - For Railway deployment, you'll need to store models externally (e.g., Google Drive, AWS S3, Hugging Face Hub)
   - Update `app.py` to download models from external storage on startup
   - Or use Railway's persistent storage volumes

7. **Deploy**
   - Railway will automatically build and deploy
   - Your app will be available at `https://YOUR_PROJECT.railway.app`

### Deploying to Other Platforms

**Streamlit Cloud**:
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Deploy (note: model files need to be hosted separately)

**Heroku**:
1. Create `Procfile`: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
2. Create `runtime.txt`: `python-3.10.0`
3. Deploy via Heroku CLI or GitHub integration

**AWS/GCP/Azure**:
- Use container services (ECS, Cloud Run, Container Instances)
- Create Dockerfile with Streamlit
- Store models in cloud storage (S3, GCS, Blob Storage)

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t intent-app .
docker run -p 8501:8501 intent-app
```

## Project Structure

```
intent_project/
├── app.py                          # Main Streamlit web application
├── generate_visualizations.py      # Script for generating evaluation visualizations
├── generate_slot_filling_flowchart.py  # Script for generating slot filling flowchart
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── SETUP_GITHUB.md                 # GitHub setup instructions
├── hyperparameters_report.md       # Detailed training hyperparameters
├── hyperparameters_table.md        # Hyperparameters in table format
│
├── NLP_IntentClassification.ipynb  # Intent classification model training notebook
├── NLP_SlotFilling_1st.ipynb      # Slot filling model training notebook
├── NLP_CombinedIntent+SlotFilling.ipynb  # Combined evaluation notebook
├── CombinedIntent_NERSlotCRF.ipynb  # CRF-based slot filling notebook
├── SlotFilling_2.ipynb            # Additional slot filling experiments
├── SlotFilling_CRFvsNonCRF.ipynb  # CRF vs non-CRF comparison
├── CrossLingual_Evaluation.ipynb   # Cross-lingual evaluation notebook
├── CombinedIntent+SlotFilling_Evaluation.ipynb  # Combined evaluation
│
├── xlm-roberta-intent-classifier-final/  # Trained intent classification model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer files
│   ├── id2intent.json
│   ├── intent2id.json
│   └── evaluation results
│
├── slot_filling_model_crf/        # CRF-based slot filling model
│   └── final_model/
│       ├── config.json
│       ├── model files
│       └── tokenizer files
│
├── slot_filling_model/            # Alternative slot filling models
├── slot_filling_model_plaintext/  # Plaintext slot filling model
│
├── data/                          # MASSIVE dataset files (51 languages)
│   ├── en-US.jsonl
│   ├── es-ES.jsonl
│   └── ... (49 more languages)
│
├── splits/                        # Data splits for evaluation
│   ├── train_data.json
│   ├── val_data.json
│   ├── test_data.json
│   ├── zero_shot_test_data.json
│   ├── few_shot_eval.json
│   └── few_shot_support.json
│
├── visualizations/                # Generated evaluation visualizations
│   ├── overall_metrics.png
│   ├── per_language_accuracy.png
│   ├── per_intent_f1.png
│   ├── slot_filling_flowchart.png
│   └── ...
│
└── tokenized_dataset/             # Preprocessed tokenized datasets
    ├── train/
    ├── validation/
    └── test/
```

## Model Performance

### Intent Classification
- Overall metrics available in `xlm-roberta-intent-classifier-final/evaluation_summary.json`
- Per-language accuracy: `xlm-roberta-intent-classifier-final/per_language_accuracy.csv`
- Per-intent F1 scores: `xlm-roberta-intent-classifier-final/per_class_f1_scores.csv`
- Classification report: `xlm-roberta-intent-classifier-final/classification_report.txt`
- Confusion matrix visualizations included

### Slot Filling
- Overall F1 Score and metrics: `slot_filling_model_plaintext/test_results.json`
- Supports 55 slot types with BIO tagging
- CRF-enhanced model for improved sequence labeling accuracy

## Training

### Training the Intent Classification Model

Use the notebook `NLP_IntentClassification.ipynb`:
1. Load and preprocess MASSIVE dataset
2. Fine-tune XLM-RoBERTa-base for sequence classification
3. Train for 5 epochs with the hyperparameters specified in `hyperparameters_report.md`
4. Evaluate on validation and test sets
5. Save best model based on F1-macro score

### Training the Slot Filling Model

Use the notebook `NLP_SlotFilling_1st.ipynb`:
1. Load and preprocess MASSIVE dataset with slot annotations
2. Convert to BIO tagging scheme
3. Fine-tune XLM-RoBERTa-base with CRF layer
4. Train for 3 epochs with the hyperparameters specified in `hyperparameters_report.md`
5. Evaluate on validation and test sets
6. Save best model based on F1 score

### Key Training Details

**Data Preprocessing**:
- Tokenization: SentencePiece BPE (XLM-RoBERTa tokenizer)
- Truncation: Enabled (max length: 128 tokens)
- Padding: Max length padding
- Word Splitting: Subword tokenization with word alignment for slot filling

**Hardware Requirements**:
- GPU recommended (CUDA-capable)
- 16GB+ GPU memory for training
- Mixed precision (FP16) used for efficiency

**Regularization**:
- Weight Decay: 0.01 (both models)
- Dropout: 0.1 (both models)
- Early Stopping: Enabled for intent classification (patience: 2 epochs)

## Evaluation

### Generating Visualizations

Run the visualization script:
```bash
python generate_visualizations.py --model-dir xlm-roberta-intent-classifier-final/
```

This generates:
- Overall evaluation metrics (Accuracy, Micro/Macro F1)
- Per-language accuracy (lowest 10 vs highest 10)
- Per-intent F1 scores (lowest 10 vs highest 10)
- Intent distribution across test set
- Slot tag frequency analysis

### Evaluation Notebooks

- `CrossLingual_Evaluation.ipynb`: Zero-shot and few-shot cross-lingual evaluation
- `CombinedIntent+SlotFilling_Evaluation.ipynb`: Combined intent and slot evaluation
- `SlotFilling_CRFvsNonCRF.ipynb`: Comparison between CRF and non-CRF models

## Supported Languages

The system supports 51 languages from the MASSIVE dataset:

**European Languages**: English (en-US), Spanish (es-ES), French (fr-FR), German (de-DE), Italian (it-IT), Portuguese (pt-PT), Dutch (nl-NL), Polish (pl-PL), Russian (ru-RU), Swedish (sv-SE), Norwegian (nb-NO), Danish (da-DK), Finnish (fi-FI), Greek (el-GR), Romanian (ro-RO), Czech (cs-CZ), Hungarian (hu-HU), Bulgarian (bg-BG), Croatian (hr-HR), Slovak (sk-SK), Slovenian (sl-SL), Estonian (et-EE), Latvian (lv-LV), Lithuanian (lt-LT), Irish (ga-IE), Welsh (cy-GB), Icelandic (is-IS), Albanian (sq-AL)

**Asian Languages**: Chinese Simplified (zh-CN), Chinese Traditional (zh-TW), Japanese (ja-JP), Korean (ko-KR), Hindi (hi-IN), Bengali (bn-BD), Tamil (ta-IN), Telugu (te-IN), Kannada (kn-IN), Malayalam (ml-IN), Thai (th-TH), Vietnamese (vi-VN), Indonesian (id-ID), Javanese (jv-ID), Malay (ms-MY), Tagalog (tl-PH), Khmer (km-KH), Myanmar (my-MM), Mongolian (mn-MN), Georgian (ka-GE), Armenian (hy-AM)

**Middle Eastern & African Languages**: Arabic (ar-SA), Hebrew (he-IL), Persian (fa-IR), Turkish (tr-TR), Urdu (ur-PK), Swahili (sw-KE), Afrikaans (af-ZA), Amharic (am-ET), Azerbaijani (az-AZ)

See the `data/` directory for the complete list of language files.

## Intent Classes

The system classifies utterances into 60 intent classes including:
- Calendar operations (calendar_set, calendar_query, calendar_remove)
- Music and media (play_music, play_radio, play_audiobook, play_podcasts, music_query)
- Email operations (email_sendemail, email_query, email_addcontact)
- IoT device control (iot_hue_lighton, iot_hue_lightoff, iot_wemo_on, iot_wemo_off)
- Weather and news (weather_query, news_query)
- Q&A (qa_factoid, qa_definition, qa_currency)
- Lists management (lists_createoradd, lists_query, lists_remove)
- Alarms and reminders (alarm_set, alarm_query, alarm_remove)
- And many more...

See `xlm-roberta-intent-classifier-final/id2intent.json` for the complete list.

## Slot Types

The system extracts 55 different slot types including:
- Temporal: DATE_TIME, TIME, DATE, TIME_RELATIVE
- Location: LOCATION, COUNTRY, CITY
- Media: MUSIC_GENRE, ARTIST_NAME, SONG_NAME, ALBUM_NAME, APP_NAME
- Personal: PERSON, CONTACT_NAME
- Numeric: NUMBER, CURRENCY, PERCENTAGE
- And many more...

See `slot_filling_model/id2slot.json` for the complete list of slot types.

## Technical Details

### Model Architecture
- **Base Model**: XLM-RoBERTa-base (xlm-roberta-base from Hugging Face)
- **Intent Classifier**: XLM-RoBERTa + Linear classification head (768 → 60)
- **Slot Filler**: XLM-RoBERTa + Linear projection (768 → 111) + CRF layer

### Inference Pipeline
1. **Text Input**: User utterance (can be in any of 51 supported languages)
2. **Intent Classification**:
   - Tokenize with XLM-RoBERTa tokenizer
   - Pass through fine-tuned XLM-RoBERTa model
   - Apply softmax to get probability distribution
   - Return top intent with confidence score
3. **Slot Filling**:
   - Split utterance into words
   - Tokenize with word-level alignment
   - Pass through fine-tuned XLM-RoBERTa model
   - Apply CRF decoding (Viterbi algorithm)
   - Map subword predictions to word-level BIO tags
   - Extract slot spans and values

### Performance Optimizations
- Mixed precision inference (FP16) when GPU available
- Model caching to avoid reloading
- Batch processing support (though UI processes one at a time)
- CPU fallback for environments without GPU

## Dependencies

Key dependencies (see `requirements.txt` for complete list):
- `streamlit>=1.28.0`: Web application framework
- `torch>=2.0.0`: PyTorch for model inference
- `transformers>=4.30.0`: Hugging Face transformers library
- `safetensors>=0.3.0`: Safe model loading
- `numpy>=1.24.0`: Numerical operations
- `matplotlib>=3.7.0`: Visualization generation
- `pandas>=2.0.0`: Data manipulation
- `scikit-learn>=1.3.0`: Evaluation metrics
- `datasets>=2.14.0`: Dataset handling
- `accelerate>=0.20.0`: Model acceleration
- `seqeval>=1.2.2`: Sequence evaluation metrics

## File Descriptions

### Core Application Files
- `app.py`: Main Streamlit application with UI and inference logic
- `generate_visualizations.py`: Script to generate evaluation visualizations
- `generate_slot_filling_flowchart.py`: Script to generate slot filling process flowchart

### Training Notebooks
- `NLP_IntentClassification.ipynb`: Complete intent classification training pipeline
- `NLP_SlotFilling_1st.ipynb`: Slot filling model training with CRF
- `NLP_CombinedIntent+SlotFilling.ipynb`: Combined training and evaluation
- `CombinedIntent_NERSlotCRF.ipynb`: CRF-based slot filling experiments
- `SlotFilling_2.ipynb`: Additional slot filling experiments
- `SlotFilling_CRFvsNonCRF.ipynb`: Comparison study between CRF and non-CRF models
- `CrossLingual_Evaluation.ipynb`: Zero-shot and few-shot cross-lingual evaluation
- `CombinedIntent+SlotFilling_Evaluation.ipynb`: Combined evaluation metrics

### Documentation
- `README.md`: This comprehensive project documentation
- `hyperparameters_report.md`: Detailed hyperparameter documentation
- `hyperparameters_table.md`: Concise hyperparameter table
- `SETUP_GITHUB.md`: GitHub repository setup instructions

## Future Improvements

Potential enhancements:
- Support for more languages
- Real-time model fine-tuning interface
- Batch processing API
- Model versioning and A/B testing
- Integration with voice assistants
- Multi-turn conversation support
- Confidence threshold customization
- Export results to various formats

## Troubleshooting

### Common Issues

**Model Loading Errors**:
- Ensure model files are in correct directories
- Check that model paths in `app.py` are correct
- Verify sufficient disk space and memory

**CUDA Out of Memory**:
- Reduce batch size in inference
- Use CPU mode if GPU memory is limited
- Enable mixed precision (already enabled)

**Slow Inference**:
- Use GPU if available
- Ensure models are loaded once and cached
- Check system resources

**Deployment Issues**:
- Ensure all dependencies are in `requirements.txt`
- Check that port configuration matches platform requirements
- Verify model files are accessible (may need external storage)

## License

This project is for research and educational purposes.

## Acknowledgments

- **MASSIVE Dataset**: Multilingual dataset for NLU tasks (Amazon Science)
- **XLM-RoBERTa**: Base transformer model from Hugging Face
- **Hugging Face Transformers**: Model training and inference framework
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework

## Contact

For questions, issues, or contributions, please open an issue in the repository.

## Citation

If you use this project in your research, please cite:
- MASSIVE Dataset
- XLM-RoBERTa paper
- Hugging Face Transformers library
