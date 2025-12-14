# How to Get Your Hugging Face Token

## Step-by-Step Instructions

1. **Go to Hugging Face Settings**
   - Direct link: https://huggingface.co/settings/tokens
   - Or navigate: Hugging Face website → Your profile (top right) → Settings → Access Tokens

2. **Create a New Token**
   - Click the **"New token"** button
   - Give it a name (e.g., "Streamlit Deployment" or "Model Upload")
   - Select **"Write"** as the token type (you need write access to upload models)
   - Click **"Generate a token"**

3. **Copy the Token**
   - The token will appear on screen (starts with `hf_...`)
   - **IMPORTANT**: Copy it immediately! You won't be able to see it again
   - It looks like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

4. **Use the Token**
   - When you run `hf auth login` or `huggingface-cli login` in your terminal
   - Paste the token when prompted (it won't show as you type for security)
   - Press Enter

## Visual Guide

```
Hugging Face Website
  ↓
Click your profile icon (top right)
  ↓
Click "Settings"
  ↓
Click "Access Tokens" (left sidebar)
  ↓
Click "New token"
  ↓
Fill in:
  - Name: "Streamlit Deployment"
  - Type: "Write"
  ↓
Click "Generate a token"
  ↓
COPY THE TOKEN (you won't see it again!)
```

## Security Notes

- **Never commit tokens to git** - they're like passwords
- **Don't share tokens** publicly
- If you lose a token, you can delete it and create a new one
- You can have multiple tokens for different purposes

## Troubleshooting

**"Token not found" error:**
- Make sure you copied the entire token (starts with `hf_`)
- Check for extra spaces before/after the token
- Try creating a new token

**"Permission denied" error:**
- Make sure you selected "Write" token type
- Check that you're logged into the correct Hugging Face account

