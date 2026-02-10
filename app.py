import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
import json
import os
import traceback


class SimpleCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

    def forward(self, emissions, tags, mask=None, reduction="mean"):
        raise RuntimeError("SimpleCRF does not support training in this deployment.")

    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        mask = mask.bool()

        emissions = emissions.float()
        batch_size, seq_length, num_tags = emissions.size()

        score = self.start_transitions + emissions[:, 0]
        history = []

        for t in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_transitions = self.transitions.unsqueeze(0)
            next_score = broadcast_score + broadcast_transitions
            best_score, best_tag = next_score.max(dim=1)
            best_score = best_score + emissions[:, t]
            history.append(best_tag)

            mask_t = mask[:, t].unsqueeze(1)
            score = torch.where(mask_t, best_score, score)

        score = score + self.end_transitions
        best_score, best_last_tag = score.max(dim=1)

        seq_ends = mask.long().sum(dim=1) - 1
        best_paths = []
        for i in range(batch_size):
            seq_end = seq_ends[i].item()
            best_tag = best_last_tag[i].item()
            best_path = [best_tag]

            for history_t in reversed(history[:seq_end]):
                best_tag = history_t[i][best_path[-1]].item()
                best_path.append(best_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths


class XLMRobertaWithCRF(nn.Module):
    def __init__(self, model_name, num_labels, id2label, label2id):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
        self.crf = SimpleCRF(num_labels)
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            labels = torch.where(labels == -100, torch.zeros_like(labels), labels)
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction="mean")
            return {"loss": loss, "logits": logits}

        predictions = self.crf.decode(logits, mask=attention_mask.bool())
        return {"logits": logits, "predictions": predictions}

st.set_page_config(page_title="Multilingual NLU System", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #f0f4ff 0%, #e8f4f8 40%, #f5f9ff 100%);
        padding-top: 0.5rem;
        padding-bottom: 1.5rem;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 960px;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
        gap: 0.25rem;
    }
    div[data-testid="stVerticalBlock"]:has(div.input-box) + div[data-testid="stVerticalBlock"] {
        margin-top: 0.25rem !important;
    }
    .element-container:has(div.input-box) {
        margin-bottom: 0.25rem !important;
    }
    .element-container:has(div.result-card) {
        margin-top: 0.25rem !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2847 0%, #0b1f3a 100%);
        color: #f7f9ff;
        padding-bottom: 2rem !important;
    }
    [data-testid="stSidebar"] > [data-testid="stVerticalBlock"] {
        padding-bottom: 1.5rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        padding: 0.85rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMetric, [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stCaption {
        color: #f7f9ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: rgba(247, 249, 255, 0.88) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #fff !important;
    }
    [data-testid="stSidebar"] ul, [data-testid="stSidebar"] li {
        color: #f7f9ff !important;
    }
    .hero-card {
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(20px);
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(12, 63, 138, 0.06);
        box-shadow: 0 4px 24px rgba(15, 55, 120, 0.08);
        margin-bottom: 1.25rem;
    }
    .hero-card h1 {
        font-size: 2.1rem;
        margin-bottom: 0.35rem;
        margin-top: 0.25rem;
        color: #0b1f3a;
        font-weight: 700;
    }
    .hero-card p {
        margin-bottom: 0.25rem;
        margin-top: 0.25rem;
        font-size: 1.05rem;
        line-height: 1.55;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(46,110,223,0.15), rgba(134,231,214,0.12));
        color: #0b1f3a;
        font-weight: 600;
        letter-spacing: 0.02em;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .input-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        border: 1px solid rgba(12, 63, 138, 0.06);
        box-shadow: 0 2px 16px rgba(15, 55, 120, 0.06);
        margin-bottom: 0.35rem;
    }
    [data-baseweb="textarea"] {
        border: none !important;
        box-shadow: none !important;
    }
    [data-baseweb="textarea"] textarea {
        border: 1px solid rgba(46,110,223,0.2) !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
        font-size: 1.05rem !important;
    }
    div[data-baseweb="textarea"] {
        border: none !important;
        box-shadow: none !important;
    }
    [data-baseweb="textarea"] textarea:focus {
        border-color: rgba(46,110,223,0.5) !important;
        box-shadow: 0 0 0 2px rgba(46,110,223,0.15) !important;
        outline: none !important;
    }
    div.stButton>button {
        border: 1px solid rgba(46,110,223,0.3);
        border-radius: 12px;
        background: rgba(255,255,255,0.9);
        color: #0b1f3a;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    div.stButton>button:hover {
        background: rgba(46,110,223,0.08);
        transform: translateY(-1px);
        border-color: rgba(46,110,223,0.45);
    }
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(12, 63, 138, 0.06);
        box-shadow: 0 4px 20px rgba(15, 55, 120, 0.06);
        margin-bottom: 0.75rem;
        margin-top: 0.75rem;
    }
    .slot-tag {
        display: inline-flex;
        flex-direction: column;
        gap: 0.25rem;
        padding: 0.65rem 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(46,110,223,0.08), rgba(134,231,214,0.06));
        border: 1px solid rgba(46,110,223,0.15);
        margin: 0.35rem 0.35rem 0 0;
        min-width: 140px;
    }
    .slot-tag span {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: rgba(11,31,58,0.65);
    }
    .slot-tag strong {
        font-size: 1.1rem;
        color: #0b1f3a;
    }
    .footer-note {
        color: rgba(11,31,58,0.6);
        font-size: 0.75rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .stSubheader { margin-bottom: 0.15rem; margin-top: 0.15rem; }
    .stCaption { margin-bottom: 0.15rem; margin-top: 0.05rem; }
    [data-baseweb="textarea"] { min-height: 130px; }
    label[for*="utterance"] { font-weight: 600 !important; font-size: 1.05rem !important; }
    div[data-testid="stMetricLabel"] { font-weight: 600 !important; }
    textarea[data-baseweb="textarea"], div[data-baseweb="textarea"] textarea,
    [data-baseweb="textarea"] textarea { font-weight: 500 !important; }
    div[data-testid="stTabs"] { margin-top: 0.25rem; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.95rem !important; font-weight: 600 !important; }
    [data-testid="stMetric"] { font-size: 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-pill">🌍 Multilingual AI Assistant</div>
            <h1>Intent Detection &amp; Slot Filling</h1>
            <p style="max-width: 640px; color: rgba(11,31,58,0.72); font-size: 1.05rem; line-height: 1.55; margin: 0.4rem 0 0 0;">
                Understand user utterances across 51 languages with a combined XLM-RoBERTa pipeline—
                intent classification and CRF-enhanced slot extraction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "utterance" not in st.session_state:
    st.session_state.utterance = ""

example_utterances = [
    "Wake me up at 6 AM tomorrow",
    "Book a table at an Italian restaurant for 7pm",
    "Play jazz music by Miles Davis",
    "What's the weather in Paris next weekend?",
]

with st.sidebar:
    st.metric("Supported Languages", "51")
    st.metric("Intent Classes", "60")
    st.metric("Slot Types", "55")

    st.markdown("---")
    st.markdown(
        """
        - Try mixing languages and dialects  
        - Experiment with compound commands  
        - Keep utterances conversational
        """.strip()
    )

    st.markdown("---")
    st.caption("Built with XLM-RoBERTa and CRF-based slot tagging.")


def _set_example(example_value):
    st.session_state.utterance = example_value

def _get_secret_or_env(section, key, env_var):
    """
    Helper to read a value from Streamlit secrets (section/key) or an env var.
    Returns None if not found.
    """
    # Try Streamlit secrets first
    try:
        if section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        # st.secrets may not be configured locally
        pass

    # Fallback to environment variable
    return os.getenv(env_var)


def _load_tokenizer(path_or_repo):
    """Load tokenizer; use fix_mistral_regex=False if supported to avoid false-positive warning."""
    try:
        return AutoTokenizer.from_pretrained(path_or_repo, fix_mistral_regex=False)
    except TypeError:
        return AutoTokenizer.from_pretrained(path_or_repo)


@st.cache_resource
def load_models():
    """
    Load models either from local directories (for local dev)
    or from Hugging Face Hub (for Streamlit Cloud / remote).

    Priority:
    1. If HF repo IDs are provided via secrets or env vars, load from Hub
    2. Else, if local model directories exist, load from disk
    3. Otherwise, show a clear error with deployment instructions
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read HF repo IDs from Streamlit secrets or env vars
    intent_hf_repo = _get_secret_or_env("models", "intent_hf_repo", "INTENT_MODEL_HF_REPO")
    slot_hf_repo = _get_secret_or_env("models", "slot_hf_repo", "SLOT_MODEL_HF_REPO")

    # Initialize variables
    intent_tokenizer = None
    intent_model = None
    id2intent = None
    slot_tokenizer = None
    slot_model = None
    id2slot = None

    try:
        # Case 1: Load from Hugging Face Hub if repo IDs are configured
        if intent_hf_repo and slot_hf_repo:
            # Download full snapshots so we can also read metadata JSON files
            intent_repo_dir = snapshot_download(repo_id=intent_hf_repo)
            slot_repo_dir = snapshot_download(repo_id=slot_hf_repo)

            # Intent model + tokenizer
            intent_tokenizer = _load_tokenizer(intent_repo_dir)
            intent_model = AutoModelForSequenceClassification.from_pretrained(
                intent_repo_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(device)
            intent_model.eval()

            # Load intent label mappings
            with open(os.path.join(intent_repo_dir, "id2intent.json"), "r") as f:
                id2intent = json.load(f)

            # Slot tokenizer + CRF model
            slot_tokenizer = _load_tokenizer(slot_repo_dir)

            with open(os.path.join(slot_repo_dir, "id2label.json"), "r") as f:
                id2slot = json.load(f)
            with open(os.path.join(slot_repo_dir, "label2id.json"), "r") as f:
                slot2id = json.load(f)

            slot_model = XLMRobertaWithCRF(
                model_name="xlm-roberta-base",
                num_labels=len(slot2id),
                id2label=id2slot,
                label2id=slot2id,
            ).to(device)

            model_state = load_file(os.path.join(slot_repo_dir, "model.safetensors"))
            slot_model.load_state_dict(model_state)
            slot_model.eval()

        else:
            # Case 2: Local development - load from local model directories
            intent_model_dir = os.path.join(current_dir, "xlm-roberta-intent-classifier-final")
            slot_model_dir = os.path.join(current_dir, "slot_filling_model_crf", "final_model")

            if not os.path.isdir(intent_model_dir) or not os.path.isdir(slot_model_dir):
                raise RuntimeError(
                    "Models not found.\n\n"
                    "- For local dev: ensure 'xlm-roberta-intent-classifier-final/' and "
                    "'slot_filling_model_crf/final_model/' exist next to app.py.\n"
                    "- For Streamlit Cloud: upload models to Hugging Face Hub and set "
                    "'models.intent_hf_repo' and 'models.slot_hf_repo' in Streamlit secrets "
                    "or INTENT_MODEL_HF_REPO / SLOT_MODEL_HF_REPO env vars."
                )

            # Intent model + tokenizer from local disk
            intent_tokenizer = _load_tokenizer(intent_model_dir)
            intent_model = AutoModelForSequenceClassification.from_pretrained(
                intent_model_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(device)
            intent_model.eval()

            with open(os.path.join(intent_model_dir, "id2intent.json"), "r") as f:
                id2intent = json.load(f)

            # Slot tokenizer + CRF model from local disk
            slot_tokenizer = _load_tokenizer(slot_model_dir)

            with open(os.path.join(slot_model_dir, "id2label.json"), "r") as f:
                id2slot = json.load(f)
            with open(os.path.join(slot_model_dir, "label2id.json"), "r") as f:
                slot2id = json.load(f)

            slot_model = XLMRobertaWithCRF(
                model_name="xlm-roberta-base",
                num_labels=len(slot2id),
                id2label=id2slot,
                label2id=slot2id,
            ).to(device)

            model_state = load_file(os.path.join(slot_model_dir, "model.safetensors"))
            slot_model.load_state_dict(model_state)
            slot_model.eval()

    except Exception as e:
        tb = traceback.format_exc()
        st.error(
            "Error loading models.\n\n"
            f"**Details:** {str(e)}\n\n"
            "**Traceback:**\n```\n" + tb + "\n```\n\n"
            "On Streamlit Cloud, make sure you have:\n"
            "- Uploaded the models to Hugging Face Hub, and\n"
            "- Set 'models.intent_hf_repo' and 'models.slot_hf_repo' in Streamlit secrets."
        )
        st.stop()

    return {
        "intent_tokenizer": intent_tokenizer,
        "intent_model": intent_model,
        "id2intent": id2intent,
        "device": device,
        "slot_tokenizer": slot_tokenizer,
        "slot_model": slot_model,
        "id2slot": id2slot,
    }

# Show loading indicator
with st.spinner("Loading models... This may take a few minutes for the first time."):
    models = load_models()

@torch.inference_mode()
def predict_intent(utterance):
    try:
        device = models['device']
        tokenizer = models['intent_tokenizer']
        model = models['intent_model']
        id2intent = models['id2intent']
        
        inputs = tokenizer(utterance, return_tensors='pt', truncation=True, 
                          max_length=128, padding='max_length').to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).squeeze()[pred_id].item()
        
        return id2intent[str(pred_id)], confidence
    except Exception as e:
        st.error(f"Error in intent prediction: {str(e)}")
        return "Error", 0.0

@torch.inference_mode()
def extract_slots(utterance):
    try:
        device = models['device']
        tokenizer = models['slot_tokenizer']
        model = models['slot_model']
        id2slot = models['id2slot']
        
        words = utterance.strip().split()
        if not words:
            return []

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        word_ids = encoding.word_ids(batch_index=0)
        inputs = {k: v.to(device) for k, v in encoding.items()}

        autocast_enabled = torch.cuda.is_available() and device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

        predictions = outputs['predictions'][0]

        word_predictions = []
        last_word_idx = -1
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != last_word_idx:
                word_predictions.append(predictions[token_idx])
                last_word_idx = word_idx

        slots = []
        current_slot_type = None
        current_slot_words = []

        for idx, word in enumerate(words):
            if idx >= len(word_predictions):
                break

            pred_id = int(word_predictions[idx])
            slot_label = id2slot[str(pred_id)]

            if slot_label == 'O':
                if current_slot_type and current_slot_words:
                    slots.append((current_slot_type, ' '.join(current_slot_words)))
                current_slot_type = None
                current_slot_words = []
            elif slot_label.startswith('B-'):
                if current_slot_type and current_slot_words:
                    slots.append((current_slot_type, ' '.join(current_slot_words)))
                current_slot_type = slot_label[2:]
                current_slot_words = [word]
            elif slot_label.startswith('I-'):
                slot_type = slot_label[2:]
                if slot_type == current_slot_type:
                    current_slot_words.append(word)
                else:
                    if current_slot_type and current_slot_words:
                        slots.append((current_slot_type, ' '.join(current_slot_words)))
                    current_slot_type = slot_type
                    current_slot_words = [word]

        if current_slot_type and current_slot_words:
            slots.append((current_slot_type, ' '.join(current_slot_words)))
        
        return slots
    except Exception as e:
        st.error(f"Error in slot extraction: {str(e)}")
        return []

with st.container():
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    input_cols = st.columns([3, 1.5])

    with input_cols[0]:
        utterance = st.text_area(
            "Utterance",
            value=st.session_state.get("utterance", ""),
            height=100,
            key="utterance",
            label_visibility="visible"
        )

    with input_cols[1]:
        st.markdown("**Example Utterances**")
        example_options = ["Select an example..."] + example_utterances
        
        selected_example = st.selectbox(
            "Choose an example",
            example_options,
            key="example_select",
            label_visibility="collapsed"
        )
        if selected_example and selected_example != "Select an example...":
            st.session_state.utterance = selected_example
    st.markdown("</div>", unsafe_allow_html=True)

utterance_value = st.session_state.get("utterance", "").strip()

if utterance_value:
    intent, confidence = predict_intent(utterance_value)
    slots = extract_slots(utterance_value)

    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Analysis Summary")
        
        # Intent and Confidence in one row
        intent_cols = st.columns([1.5, 1])
        with intent_cols[0]:
            st.metric("Detected Intent", intent, label_visibility="visible")
        with intent_cols[1]:
            st.metric("Confidence", f"{confidence:.2%}", label_visibility="visible")
        
        st.progress(confidence)
        
        # Slots directly below
        st.markdown("**Extracted Slots:**")
        if slots:
            slots_html = ''.join([
                f'<div class="slot-tag"><span>{slot_type}</span><strong>{slot_value}</strong></div>'
                for slot_type, slot_value in slots
            ])
            st.markdown(slots_html, unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: rgba(11,31,58,0.6); font-size: 0.9rem; padding: 0.5rem 0;'>No slot values detected.</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <p class="footer-note">
    XLM-RoBERTa + CRF · MASSIVE Dataset
    </p>
    """,
    unsafe_allow_html=True,
)