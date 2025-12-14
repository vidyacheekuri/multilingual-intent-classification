# Deploying to Streamlit Cloud

Streamlit Cloud has limitations on repository size (1GB free tier). Since the trained models are large, you have several options:

## Option 1: Upload Models to Hugging Face Hub (Recommended)

1. **Create a Hugging Face account** at https://huggingface.co
2. **Create model repositories**:
   - Create a repository for intent classifier: `your-username/xlm-roberta-intent-classifier`
   - Create a repository for slot filling: `your-username/xlm-roberta-slot-filling-crf`
3. **Upload your models**:
   ```bash
   # Install huggingface_hub
   pip install huggingface_hub
   
   # Login
   huggingface-cli login
   
   # Upload intent classifier
   cd xlm-roberta-intent-classifier-final
   huggingface-cli upload your-username/xlm-roberta-intent-classifier .
   
   # Upload slot filling model
   cd ../slot_filling_model_crf/final_model
   huggingface-cli upload your-username/xlm-roberta-slot-filling-crf .
   ```
4. **Update app.py** to load from Hugging Face Hub instead of local paths

## Option 2: Download Models from Cloud Storage on Startup

If your models are stored in Google Drive or another cloud storage:

1. **Create a download script** that runs before the app starts
2. **Store models in a publicly accessible location** (or use service account)
3. **Download models on first run** and cache them

## Option 3: Use Streamlit Secrets for Model Paths

1. Go to your Streamlit Cloud app settings
2. Add secrets for model storage URLs or Hugging Face repo IDs
3. Update app.py to read from secrets

## Option 4: Use a Different Platform

Consider these alternatives that support larger files:
- **Railway**: 500MB free, then pay-as-you-go
- **Render**: Free tier with 512MB RAM
- **Fly.io**: Generous free tier
- **AWS/GCP/Azure**: More control, pay-as-you-go

## Quick Fix: Update app.py for Hugging Face Hub

If you upload models to Hugging Face Hub, update the model loading in `app.py`:

```python
# Replace local paths with Hugging Face Hub IDs
intent_model_dir = "your-username/xlm-roberta-intent-classifier"
slot_model_dir = "your-username/xlm-roberta-slot-filling-crf"
```

Then remove `local_files_only=True` from the `from_pretrained()` calls.

