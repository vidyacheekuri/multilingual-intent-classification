# Deploying to Streamlit Cloud

Streamlit Cloud has limitations on repository size (1GB free tier). Since the trained models are large, you have several options:

## Option 1: Upload Models to Hugging Face Hub (Recommended)

### Quick Method (Automated Script)

1. **Create a Hugging Face account** at https://huggingface.co
2. **Login to Hugging Face**:
   ```bash
   hf auth login
   # Or use: huggingface-cli login (deprecated but still works)
   ```
   Enter your token from https://huggingface.co/settings/tokens

3. **Run the upload script**:
   ```bash
   python upload_models_to_hf.py --username YOUR_USERNAME
   ```
   
   This will automatically:
   - Create the repositories on Hugging Face Hub
   - Upload the intent classification model
   - Upload the slot filling model

4. **Configure Streamlit Cloud secrets**:
   - Go to your Streamlit Cloud app settings
   - Click "Secrets" tab
   - Add these secrets:
     ```toml
     [models]
     intent_hf_repo = "YOUR_USERNAME/xlm-roberta-intent-classifier"
     slot_hf_repo = "YOUR_USERNAME/xlm-roberta-slot-filling-crf"
     ```
   - Or set as environment variables in Streamlit Cloud

5. **Redeploy your app** - it will automatically load from Hugging Face Hub!

### Manual Method

If you prefer to upload manually:

1. **Create a Hugging Face account** at https://huggingface.co
2. **Login**:
   ```bash
   hf auth login
   ```
3. **Create repositories** (or they'll be created automatically):
   - `your-username/xlm-roberta-intent-classifier`
   - `your-username/xlm-roberta-slot-filling-crf`
4. **Upload models**:
   ```bash
   # Upload intent classifier
   cd xlm-roberta-intent-classifier-final
   huggingface-cli upload your-username/xlm-roberta-intent-classifier .
   
   # Upload slot filling model
   cd ../slot_filling_model_crf/final_model
   huggingface-cli upload your-username/xlm-roberta-slot-filling-crf .
   ```

**Note**: The `app.py` file already supports loading from Hugging Face Hub! Just set the environment variables or Streamlit secrets.

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

## How It Works

The `app.py` file automatically detects whether to use local files or Hugging Face Hub:

1. **Local files** (default): Looks for models in local directories
2. **Hugging Face Hub**: If environment variables or Streamlit secrets are set:
   - `INTENT_MODEL_HF_REPO` or `models.intent_hf_repo`
   - `SLOT_MODEL_HF_REPO` or `models.slot_hf_repo`

No code changes needed! Just set the secrets and redeploy.

## Troubleshooting

### Models not loading on Streamlit Cloud
- Ensure models are uploaded to Hugging Face Hub
- Check that Streamlit secrets are set correctly
- Verify repository names match exactly (case-sensitive)
- Check Streamlit Cloud logs for detailed error messages

### Upload fails
- Ensure you're logged in: `hf auth login`
- Check that you have write access to the repositories
- Verify model directories exist locally
- Check your internet connection (uploads can be large)

