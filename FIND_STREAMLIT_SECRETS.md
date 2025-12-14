# How to Find Streamlit Cloud Secrets Tab

## Method 1: From App Dashboard

1. **Go to your Streamlit Cloud dashboard**: https://share.streamlit.io
2. **Click on your app**: "multilingual-intent-classification"
3. **Click the three dots (⋮) menu** next to your app
4. **Select "Settings"** or "Manage app"
5. **Click the "Secrets" tab** in the settings page

## Method 2: From Within Your App

1. **Open your deployed app** (click on it to view)
2. **Look for "Manage app" button** (usually in the bottom right or top right)
3. **Click "Manage app"**
4. **Click "Settings"** in the menu
5. **Click "Secrets" tab**

## Method 3: Direct URL

If you know your app name, you can go directly to:
```
https://share.streamlit.io/[your-username]/[app-name]/settings
```

For example:
```
https://share.streamlit.io/vidyacheekuri/multilingual-intent-classification/settings
```

## What to Add in Secrets

Once you're in the Secrets tab, add this in TOML format:

```toml
[models]
intent_hf_repo = "vidyac9/xlm-roberta-intent-classifier"
slot_hf_repo = "vidyac9/xlm-roberta-slot-filling-crf"
```

Or as environment variables:

```
INTENT_MODEL_HF_REPO=vidyac9/xlm-roberta-intent-classifier
SLOT_MODEL_HF_REPO=vidyac9/xlm-roberta-slot-filling-crf
```

## Visual Guide

```
Streamlit Dashboard
  ↓
Click "multilingual-intent-classification" app
  ↓
Click ⋮ (three dots menu)
  ↓
Click "Settings" or "Manage app"
  ↓
Click "Secrets" tab
  ↓
Paste your secrets
  ↓
Click "Save"
```

## Troubleshooting

**Can't find Secrets tab?**
- Make sure you're the owner of the app
- Try refreshing the page
- Check that you're logged into the correct Streamlit account

**Secrets not working?**
- Make sure the format is correct (TOML format)
- Check for typos in the repository names
- Redeploy the app after saving secrets

