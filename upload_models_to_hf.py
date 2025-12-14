#!/usr/bin/env python3
"""
Script to upload trained models to Hugging Face Hub.

Usage:
    python upload_models_to_hf.py --username YOUR_USERNAME

This will:
1. Create repositories on Hugging Face Hub
2. Upload intent classification model
3. Upload slot filling model
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import HfFolder

def upload_intent_model(username, model_dir="xlm-roberta-intent-classifier-final"):
    """Upload intent classification model to Hugging Face Hub."""
    repo_id = f"{username}/xlm-roberta-intent-classifier"
    
    print(f"\n{'='*60}")
    print(f"Uploading Intent Classification Model")
    print(f"{'='*60}")
    print(f"Repository: {repo_id}")
    print(f"Local directory: {model_dir}")
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"ERROR: Directory {model_dir} not found!")
        return False
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"ERROR creating repository: {e}")
        return False
    
    # Upload the model with retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if retry_count > 0:
                print(f"\nRetry attempt {retry_count + 1}/{max_retries}...")
            print("Uploading files... This may take a while...")
            upload_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=["*.png", "*.csv", "*.txt", "__pycache__", "*.pyc"]
            )
            print(f"✓ Successfully uploaded intent model to {repo_id}")
            return True
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"ERROR uploading model: {e}")
                print(f"Retrying in 10 seconds...")
                import time
                time.sleep(10)
            else:
                print(f"ERROR uploading model after {max_retries} attempts: {e}")
                return False

def upload_slot_model(username, model_dir="slot_filling_model_crf/final_model"):
    """Upload slot filling model to Hugging Face Hub."""
    repo_id = f"{username}/xlm-roberta-slot-filling-crf"
    
    print(f"\n{'='*60}")
    print(f"Uploading Slot Filling Model")
    print(f"{'='*60}")
    print(f"Repository: {repo_id}")
    print(f"Local directory: {model_dir}")
    
    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"ERROR: Directory {model_dir} not found!")
        return False
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"ERROR creating repository: {e}")
        return False
    
    # Upload the model with retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if retry_count > 0:
                print(f"\nRetry attempt {retry_count + 1}/{max_retries}...")
            print("Uploading files... This may take a while (especially for large model files)...")
            upload_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=["__pycache__", "*.pyc"]
            )
            print(f"✓ Successfully uploaded slot model to {repo_id}")
            return True
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"ERROR uploading model: {e}")
                print(f"Retrying in 10 seconds...")
                import time
                time.sleep(10)
            else:
                print(f"ERROR uploading model after {max_retries} attempts: {e}")
                print("\nTroubleshooting tips:")
                print("- Check your internet connection")
                print("- The model file might be very large, try uploading during off-peak hours")
                print("- You can also try uploading manually using: huggingface-cli upload")
                return False

def main():
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--intent-model-dir",
        type=str,
        default="xlm-roberta-intent-classifier-final",
        help="Path to intent classification model directory"
    )
    parser.add_argument(
        "--slot-model-dir",
        type=str,
        default="slot_filling_model_crf/final_model",
        help="Path to slot filling model directory"
    )
    parser.add_argument(
        "--intent-only",
        action="store_true",
        help="Upload only intent classification model"
    )
    parser.add_argument(
        "--slot-only",
        action="store_true",
        help="Upload only slot filling model"
    )
    
    args = parser.parse_args()
    
    # Check if user is logged in
    token = HfFolder.get_token()
    if token is None:
        print("ERROR: Not logged in to Hugging Face!")
        print("Please run: hf auth login")
        return
    
    print(f"✓ Logged in as: {HfApi().whoami()['name']}")
    
    success = True
    
    # Upload intent model
    if not args.slot_only:
        if not upload_intent_model(args.username, args.intent_model_dir):
            success = False
    
    # Upload slot model
    if not args.intent_only:
        if not upload_slot_model(args.username, args.slot_model_dir):
            success = False
    
    if success:
        print(f"\n{'='*60}")
        print("SUCCESS! All models uploaded.")
        print(f"{'='*60}")
        print("\nNext steps for Streamlit Cloud:")
        print("1. Go to your Streamlit Cloud app settings")
        print("2. Add these secrets:")
        print(f"   INTENT_MODEL_HF_REPO = {args.username}/xlm-roberta-intent-classifier")
        print(f"   SLOT_MODEL_HF_REPO = {args.username}/xlm-roberta-slot-filling-crf")
        print("3. Redeploy your app")
    else:
        print("\nSome uploads failed. Please check the errors above.")

if __name__ == "__main__":
    main()

