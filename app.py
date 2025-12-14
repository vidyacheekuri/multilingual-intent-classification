import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from safetensors.torch import load_file
import json
import os


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
        background: linear-gradient(135deg, rgba(46,110,223,0.09), rgba(134,231,214,0.18));
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
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
        background: #0b1f3a;
        color: #f7f9ff;
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
        color: rgba(247, 249, 255, 0.85) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #f7f9ff !important;
    }
    [data-testid="stSidebar"] ul, [data-testid="stSidebar"] li {
        color: #f7f9ff !important;
    }
    .hero-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(16px);
        padding: 0.5rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(12, 63, 138, 0.08);
        box-shadow: 0 15px 30px rgba(15, 55, 120, 0.1);
        margin-bottom: 0.5rem;
    }
    .hero-card h1 {
        font-size: 1.5rem;
        margin-bottom: 0.15rem;
        margin-top: 0.15rem;
        color: #0b1f3a;
    }
    .hero-card p {
        margin-bottom: 0.15rem;
        margin-top: 0.15rem;
        font-size: 0.85rem;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(46,110,223,0.12);
        color: #0b1f3a;
        font-weight: 600;
        letter-spacing: 0.01em;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }
    .input-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 0.5rem 0.75rem;
        border: 1px solid rgba(12, 63, 138, 0.08);
        box-shadow: 0 10px 20px rgba(15, 55, 120, 0.08);
        margin-bottom: 0.25rem;
    }
    div.stButton>button {
        border: 1px solid rgba(46,110,223,0.25);
        border-radius: 999px;
        background: rgba(255,255,255,0.85);
        color: #0b1f3a;
        padding: 0.25rem 0.6rem;
        font-weight: 600;
        font-size: 0.75rem;
        transition: all 0.2s ease;
        box-shadow: none;
        margin-bottom: 0.25rem;
    }
    div.stButton>button:hover {
        background: rgba(46,110,223,0.1);
        transform: translateY(-1px);
        border-color: rgba(46,110,223,0.4);
    }
    .result-card {
        background: rgba(255, 255, 255, 0.92);
        padding: 0.75rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(12, 63, 138, 0.08);
        box-shadow: 0 15px 30px rgba(15, 55, 120, 0.08);
        margin-bottom: 0.25rem;
        margin-top: 0.25rem;
    }
    .slot-tag {
        display: inline-flex;
        flex-direction: column;
        gap: 0.2rem;
        padding: 0.5rem 0.75rem;
        border-radius: 10px;
        background: rgba(46,110,223,0.1);
        border: 1px solid rgba(46,110,223,0.18);
        margin: 0.25rem 0.25rem 0 0;
        min-width: 120px;
    }
    .slot-tag span {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: rgba(11,31,58,0.7);
    }
    .slot-tag strong {
        font-size: 1rem;
        color: #0b1f3a;
    }
    .footer-note {
        color: rgba(11,31,58,0.65);
        font-size: 0.75rem;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }
    .stSubheader {
        margin-bottom: 0.15rem;
        margin-top: 0.15rem;
    }
    .stCaption {
        margin-bottom: 0.15rem;
        margin-top: 0.05rem;
    }
    [data-baseweb="textarea"] {
        min-height: 100px;
    }
    label[for*="utterance"] {
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: bold !important;
    }
    textarea[data-baseweb="textarea"] {
        font-weight: bold !important;
    }
    div[data-baseweb="textarea"] textarea {
        font-weight: bold !important;
    }
    [data-baseweb="textarea"] textarea {
        font-weight: bold !important;
    }
    div[data-testid="stTabs"] {
        margin-top: 0.25rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-pill"> Multilingual AI Assistant</div>
            <h1>Intent Detection &amp; Slot Filling</h1>
            <p style="max-width: 640px; color: rgba(11,31,58,0.75); font-size: 0.9rem; line-height: 1.4; margin: 0.25rem 0;">
                Understand user utterances across 51 languages with a combined XLM-RoBERTa pipeline.
                Powered by multilingual intent classification and CRF-enhanced slot extraction.
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

@st.cache_resource
def load_models():
    # Get the current directory (where app.py is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize variables
    intent_tokenizer = None
    intent_model = None
    id2intent = None
    slot_tokenizer = None
    slot_model = None
    id2slot = None
    
    try:
        intent_model_dir = os.path.join(current_dir, 'xlm-roberta-intent-classifier-final')
        intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_dir)
        intent_model = AutoModelForSequenceClassification.from_pretrained(
            intent_model_dir, dtype=torch.float16
        ).to(device)
        intent_model.eval()
        
        with open(os.path.join(current_dir, 'xlm-roberta-intent-classifier-final', 'intent2id.json'), 'r') as f:
            intent2id = json.load(f)
        with open(os.path.join(current_dir, 'xlm-roberta-intent-classifier-final', 'id2intent.json'), 'r') as f:
            id2intent = json.load(f)
        
        slot_model_dir = os.path.join(current_dir, 'slot_filling_model_crf', 'final_model')
        slot_tokenizer = AutoTokenizer.from_pretrained(slot_model_dir)
        
        with open(os.path.join(slot_model_dir, 'id2label.json'), 'r') as f:
            id2slot = json.load(f)
        with open(os.path.join(slot_model_dir, 'label2id.json'), 'r') as f:
            slot2id = json.load(f)

        slot_model = XLMRobertaWithCRF(
            model_name='xlm-roberta-base',
            num_labels=len(slot2id),
            id2label=id2slot,
            label2id=slot2id
        ).to(device)

        model_state = load_file(os.path.join(slot_model_dir, 'model.safetensors'))
        slot_model.load_state_dict(model_state)
        slot_model.eval()
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    return {
        'intent_tokenizer': intent_tokenizer,
        'intent_model': intent_model,
        'id2intent': id2intent,
        'device': device,
        'slot_tokenizer': slot_tokenizer,
        'slot_model': slot_model,
        'id2slot': id2slot
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