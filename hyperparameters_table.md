# Hyperparameters Summary

## Intent Classification Model

| Hyperparameter | Value |
|----------------|-------|
| **Base Model** | XLM-RoBERTa-base |
| **Learning Rate** | 2e-5 |
| **Train Batch Size** | 16 |
| **Eval Batch Size** | 32 |
| **Epochs** | 5 |
| **Weight Decay** | 0.01 |
| **Warmup Steps** | 500 |
| **Mixed Precision (FP16)** | Yes |
| **Max Sequence Length** | 128 |
| **Optimizer** | AdamW |
| **Random Seed** | 42 |
| **Number of Classes** | 60 intents |
| **Training Samples** | 230,280 |
| **Best Model Metric** | F1-macro |

---

## Slot Filling Model (CRF-based)

| Hyperparameter | Value |
|----------------|-------|
| **Base Model** | XLM-RoBERTa-base + CRF |
| **Learning Rate** | 2e-5 |
| **Train Batch Size** | 32 |
| **Eval Batch Size** | 64 |
| **Epochs** | 3 |
| **Weight Decay** | 0.01 |
| **Warmup Steps** | 200 |
| **Mixed Precision (FP16)** | Yes |
| **Max Sequence Length** | 128 |
| **Optimizer** | AdamW |
| **Random Seed** | 42 |
| **Number of Labels** | 111 (BIO tags) |
| **Slot Types** | 55 unique types |
| **Training Samples** | 230,280 |
| **Best Model Metric** | F1 score |

---

## Comparison

| Aspect | Intent Classification | Slot Filling |
|--------|----------------------|--------------|
| **Epochs** | 5 | 3 |
| **Train Batch Size** | 16 | 32 |
| **Warmup Steps** | 500 | 200 |
| **Architecture** | Sequence Classification | Token Classification + CRF |
| **Output Type** | Single label (intent) | Sequence of labels (BIO tags) |

