# Model Hyperparameters

## Intent Classification Model

### Base Model
- **Architecture**: XLM-RoBERTa-base
- **Model Size**: 768 hidden dimensions, 12 layers, 12 attention heads
- **Vocabulary Size**: 250,002 tokens
- **Number of Classes**: 60 intents

### Training Hyperparameters
- **Learning Rate**: 2e-5 (0.00002)
- **Optimizer**: AdamW (default)
- **Train Batch Size**: 16 (per device)
- **Evaluation Batch Size**: 32 (per device)
- **Number of Epochs**: 5
- **Weight Decay**: 0.01 (L2 regularization)
- **Warmup Steps**: 500
- **Learning Rate Schedule**: Linear warmup with decay
- **Mixed Precision Training**: FP16 (enabled)
- **Random Seed**: 42
- **Data Loader Workers**: 4
- **Evaluation Strategy**: End of each epoch
- **Save Strategy**: End of each epoch
- **Best Model Metric**: F1-macro
- **Save Total Limit**: 2 (keeps only 2 best checkpoints)
- **Early Stopping**: Enabled (patience: 2 epochs)

### Model Architecture Details
- **Dropout**: 0.1 (attention and hidden layers)
- **Activation Function**: GELU
- **Max Sequence Length**: 128 tokens
- **Classifier Dropout**: None (default)

### Training Data
- **Training Samples**: 230,280
- **Validation Samples**: 40,660
- **Test Samples**: 59,480
- **Number of Languages**: 20 languages
- **Total Training Steps**: 71,965

---

## Slot Filling Model (CRF-based)

### Base Model
- **Architecture**: XLM-RoBERTa-base with CRF layer
- **Model Size**: 768 hidden dimensions, 12 layers, 12 attention heads
- **Vocabulary Size**: 250,002 tokens
- **Number of Labels**: 111 (BIO tags: 55 slot types × 2 + 1 for 'O')
- **Slot Types**: 55 unique slot types

### Training Hyperparameters
- **Learning Rate**: 2e-5 (0.00002)
- **Optimizer**: AdamW (default)
- **Train Batch Size**: 32 (per device)
- **Evaluation Batch Size**: 64 (per device)
- **Number of Epochs**: 3
- **Weight Decay**: 0.01 (L2 regularization)
- **Warmup Steps**: 200
- **Learning Rate Schedule**: Linear warmup with decay
- **Mixed Precision Training**: FP16 (enabled)
- **Random Seed**: 42
- **Evaluation Strategy**: End of each epoch
- **Save Strategy**: End of each epoch
- **Best Model Metric**: F1 score
- **Save Total Limit**: 2 (keeps only 2 best checkpoints)
- **Logging Steps**: 50

### Model Architecture Details
- **CRF Layer**: Conditional Random Field for sequence labeling
- **Dropout**: 0.1 (transformer layers and classifier)
- **Activation Function**: GELU
- **Max Sequence Length**: 128 tokens
- **Classifier**: Linear layer (768 → 111 labels)
- **CRF Reduction**: Mean (for loss computation)

### Training Data
- **Training Samples**: 230,280
- **Validation Samples**: 40,660
- **Test Samples**: 59,480
- **BIO Tagging Scheme**: B- (beginning), I- (inside), O (outside)

---

## Common Training Settings

### Data Preprocessing
- **Tokenization**: SentencePiece BPE (XLM-RoBERTa tokenizer)
- **Truncation**: Enabled (max length: 128)
- **Padding**: Max length padding
- **Word Splitting**: Subword tokenization with word alignment

### Hardware & Performance
- **Device**: CUDA (GPU) when available, CPU fallback
- **Mixed Precision**: FP16 for faster training and reduced memory
- **Gradient Accumulation**: Not used (batch sizes adjusted per device)

### Regularization
- **Weight Decay**: 0.01 (both models)
- **Dropout**: 0.1 (both models)
- **Early Stopping**: Enabled for intent classification (patience: 2)

---

## Notes

1. **Intent Classification**: Trained for 5 epochs with larger warmup (500 steps) to handle class imbalance across 60 intents.

2. **Slot Filling**: Trained for 3 epochs with CRF layer to enforce valid BIO tag transitions (e.g., I-tag must follow B-tag or I-tag of same type).

3. **Batch Sizes**: Slot filling uses larger batch sizes (32/64) compared to intent classification (16/32) due to sequence-level processing efficiency.

4. **Learning Rate**: Both models use the same learning rate (2e-5), which is standard for fine-tuning transformer models.

5. **Evaluation Metrics**: 
   - Intent Classification: Accuracy, F1-macro, F1-weighted, Cohen's Kappa
   - Slot Filling: Token-level accuracy, precision, recall, F1 score

