"""
T3.2 — Devanagari Handwritten Character Recognizer
SMAI Assignment 3 · IIIT Hyderabad · 2025–26

Streamlit app with:
  • Live recognition mode  — draw a character, get top-3 predictions
  • Practice mode          — app shows a target glyph, you draw it, app scores you
  • About tab              — model info and character reference sheet

Run:
    streamlit run app.py

Dependencies:
    pip install streamlit streamlit-drawable-canvas torch torchvision Pillow opencv-python
"""

import random
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error(
        "streamlit-drawable-canvas is not installed. "
        "Run: pip install streamlit-drawable-canvas"
    )
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Devanagari Recognizer",
    page_icon="अ",
    layout="centered",
)


# ──────────────────────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Glyph display card */
    .glyph-card {
        background: #f7f7f5;
        border: 1.5px solid #e0ddd8;
        border-radius: 14px;
        text-align: center;
        padding: 18px 12px 14px;
        margin-bottom: 6px;
        color: #1a1a1a;
    }
    .glyph-big   { font-size: 72px; line-height: 1.1; }
    .glyph-label { font-size: 13px; color: #888; margin-top: 4px; }
    .glyph-rank1 { font-size: 46px; font-weight: 600; color: #1a1a1a; }
    .glyph-rank2 { font-size: 34px; color: #444; }
    .glyph-rank3 { font-size: 28px; color: #777; }

    /* Practice target card */
    .target-card {
        background: #eef4ff;
        border: 2px solid #b3ccf5;
        border-radius: 16px;
        text-align: center;
        padding: 20px 16px 16px;
        color: #1a1a1a;
    }
    .target-glyph { font-size: 96px; line-height: 1.1; }
    .target-label { font-size: 13px; color: #5a7ab8; margin-top: 6px; }

    /* Feedback banners */
    .feedback-correct {
        background: #e8f8ef;
        border-left: 5px solid #2ecc71;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        font-size: 17px;
        font-weight: 600;
        color: #1a7a45;
    }
    .feedback-wrong {
        background: #fff0f0;
        border-left: 5px solid #e74c3c;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        font-size: 17px;
        color: #9b2020;
    }
    .feedback-empty {
        background: #fdf6e3;
        border-left: 5px solid #f0c040;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        font-size: 15px;
        color: #7a6010;
    }

    .result-placeholder {
        background: #f5f5f5;
        color: #222;
        font-size: 14px;
        padding: 12px;
        border-radius: 10px;
        margin-top: 10px;
    }

    /* Score badge */
    .score-row {
        display: flex;
        gap: 14px;
        justify-content: center;
        margin: 10px 0 4px;
    }
    .score-pill {
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .pill-green  { background:#d4f5e2; color:#1a7a45; }
    .pill-red    { background:#fde0e0; color:#9b2020; }
    .pill-blue   { background:#dde8fb; color:#1a4fa0; }

    /* Character reference grid */
    .char-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(52px, 1fr));
        gap: 6px;
        margin-top: 8px;
    }
    .char-cell {
        background: #f5f4f1;
        border: 1px solid #e0ddd8;
        border-radius: 8px;
        text-align: center;
        padding: 8px 4px 5px;
        font-size: 22px;
        cursor: default;
        color: #1a1a1a;
    }
    .char-sub {
        font-size: 9px;
        color: #aaa;
        margin-top: 3px;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model definition  (matches training exactly)
# ──────────────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, num_classes=46):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32→16
        x = self.pool(F.relu(self.conv2(x)))  # 16→8
        x = self.pool(F.relu(self.conv3(x)))  # 8→4
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────────────────────
# Class data  (46 Devanagari characters, alphabetical folder order)
# ──────────────────────────────────────────────────────────────────────────────
# class_names: sorted exactly as ImageFolder sees the dataset folders
CLASS_NAMES = [
    'character_10_yna', 'character_11_taamatar', 'character_12_thaa',
    'character_13_daa', 'character_14_dhaa', 'character_15_adna',
    'character_16_tabala', 'character_17_tha', 'character_18_da',
    'character_19_dha', 'character_1_ka', 'character_20_na',
    'character_21_pa', 'character_22_pha', 'character_23_ba',
    'character_24_bha', 'character_25_ma', 'character_26_yaw',
    'character_27_ra', 'character_28_la', 'character_29_waw',
    'character_2_kha', 'character_30_motosaw', 'character_31_petchiryakha',
    'character_32_patalosaw', 'character_33_ha', 'character_34_chhya',
    'character_35_tra', 'character_36_gya', 'character_3_ga',
    'character_4_gha', 'character_5_kna', 'character_6_cha',
    'character_7_chha', 'character_8_ja', 'character_9_jha',
    'digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4',
    'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9',
]

# Rich metadata keyed by folder name → (glyph, roman)
_META = {
    "character_1_ka":              ("क",   "ka"),
    "character_2_kha":             ("ख",   "kha"),
    "character_3_ga":              ("ग",   "ga"),
    "character_4_gha":             ("घ",   "gha"),
    "character_5_kna":             ("ङ",   "nga"),
    "character_6_cha":             ("च",   "cha"),
    "character_7_chha":            ("छ",   "chha"),
    "character_8_ja":              ("ज",   "ja"),
    "character_9_jha":             ("झ",   "jha"),
    "character_10_yna":            ("ञ",   "nya"),
    "character_11_taamatar":       ("ट",   "ṭa"),
    "character_12_thaa":           ("ठ",   "ṭha"),
    "character_13_daa":            ("ड",   "ḍa"),
    "character_14_dhaa":           ("ढ",   "ḍha"),
    "character_15_adna":           ("ण",   "ṇa"),
    "character_16_tabala":         ("त",   "ta"),
    "character_17_tha":            ("थ",   "tha"),
    "character_18_da":             ("द",   "da"),
    "character_19_dha":            ("ध",   "dha"),
    "character_20_na":             ("न",   "na"),
    "character_21_pa":             ("प",   "pa"),
    "character_22_pha":            ("फ",   "pha"),
    "character_23_ba":             ("ब",   "ba"),
    "character_24_bha":            ("भ",   "bha"),
    "character_25_ma":             ("म",   "ma"),
    "character_26_yaw":            ("य",   "ya"),
    "character_27_ra":             ("र",   "ra"),
    "character_28_la":             ("ल",   "la"),
    "character_29_waw":            ("व",   "va"),
    "character_30_motosaw":        ("श",   "sha"),
    "character_31_petchiryakha":   ("ष",   "ṣha"),
    "character_32_patalosaw":      ("स",   "sa"),
    "character_33_ha":             ("ह",   "ha"),
    "character_34_chhya":          ("क्ष", "ksha"),
    "character_35_tra":            ("त्र", "tra"),
    "character_36_gya":            ("ज्ञ", "gya"),
    "digit_0":                     ("०",   "0"),
    "digit_1":                     ("१",   "1"),
    "digit_2":                     ("२",   "2"),
    "digit_3":                     ("३",   "3"),
    "digit_4":                     ("४",   "4"),
    "digit_5":                     ("५",   "5"),
    "digit_6":                     ("६",   "6"),
    "digit_7":                     ("७",   "7"),
    "digit_8":                     ("८",   "8"),
    "digit_9":                     ("९",   "9"),
}

# Index → (glyph, roman) in the same order as CLASS_NAMES
IDX_TO_GLYPH = [_META[n][0] for n in CLASS_NAMES]
IDX_TO_ROMAN = [_META[n][1] for n in CLASS_NAMES]
NUM_CLASSES  = len(CLASS_NAMES)

# Sub-groups for practice difficulty
CONSONANTS = [
    {"glyph": _META[n][0], "roman": _META[n][1], "folder": n}
    for n in CLASS_NAMES if n.startswith("character")
]
DIGITS = [
    {"glyph": _META[n][0], "roman": _META[n][1], "folder": n}
    for n in CLASS_NAMES if n.startswith("digit")
]
CLASSES_SORTED = CONSONANTS + DIGITS


# ──────────────────────────────────────────────────────────────────────────────
# Model loader  (cached so we only load once per session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(weights_path: str = "best_model.pth"):
    model = CNN(num_classes=NUM_CLASSES)
    path  = Path(weights_path)

    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        st.session_state["model_loaded"] = True
    else:
        st.session_state["model_loaded"] = False

    model.eval()
    return model


model = load_model()


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing  (canvas → model tensor)
# Uses the new app's cv2-based pipeline which matches training preprocessing
# Canvas is white-on-black; training data is also white-on-black
# ──────────────────────────────────────────────────────────────────────────────
_transform = T.Compose([
    T.Resize((32, 32)),
    T.Grayscale(1),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])

def preprocess_canvas(canvas_array: np.ndarray) -> torch.Tensor | None:
    """
    Convert raw st_canvas RGBA output to a model-ready (1,1,32,32) float tensor.
    Returns None if nothing has been drawn yet.

    Pipeline:
      1. Drop alpha → RGB → grayscale
      2. Threshold to binary (stroke vs background)
      3. Crop to bounding box of stroke
      4. Pad to 28×28, then add 2px border → 32×32 equivalent
      5. Normalise to [−1, 1]
    """
    if canvas_array is None:
        return None

    img = canvas_array[:, :, :3].astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None   # nothing drawn

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y + h, x:x + w]

    resized = cv2.resize(cropped, (28, 28))
    padded  = np.pad(resized, ((2, 2), (2, 2)), mode="constant", constant_values=0)

    pil_img = Image.fromarray(padded.astype("uint8"))
    return _transform(pil_img).unsqueeze(0)   # (1, 1, 32, 32)


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def predict(tensor: torch.Tensor, top_k: int = 3) -> list[tuple[str, str, float]]:
    """Returns list of (glyph, roman, confidence) sorted by confidence desc."""
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(top_k)
    return [
        (IDX_TO_GLYPH[i], IDX_TO_ROMAN[i], float(p))
        for i, p in zip(top_idxs.tolist(), top_probs.tolist())
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "practice_target":   None,
        "practice_score":    0,
        "practice_attempts": 0,
        "practice_streak":   0,
        "practice_history":  [],
        "canvas_key_recog":  0,
        "canvas_key_prac":   0,
        "last_result":       None,
        "model_loaded":      st.session_state.get("model_loaded", False),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: draw the canvas widget (white stroke on black — matches training data)
# ──────────────────────────────────────────────────────────────────────────────
def draw_canvas(key_suffix: str, height: int = 240) -> np.ndarray | None:
    result = st_canvas(
        fill_color       = "grey",
        stroke_width     = 8,
        stroke_color     = "white",
        background_color = "black",
        height           = height,
        width            = height,
        drawing_mode     = "freedraw",
        key              = f"canvas_{key_suffix}_{st.session_state[f'canvas_key_{key_suffix}']}",
        display_toolbar  = False,
    )
    return result.image_data if result else None


# ──────────────────────────────────────────────────────────────────────────────
# Helper: confidence bar
# ──────────────────────────────────────────────────────────────────────────────
def confidence_bar(label: str, value: float, color: str = "#4a7adf"):
    pct = round(value * 100, 1)
    st.markdown(
        f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;
                      font-size:13px;color:#555;margin-bottom:3px">
            <span>{label}</span><span>{pct}%</span>
          </div>
          <div style="background:#efefef;border-radius:6px;height:10px">
            <div style="width:{pct}%;background:{color};
                        border-radius:6px;height:10px;
                        transition:width 0.4s ease"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: pick a new practice target
# ──────────────────────────────────────────────────────────────────────────────
def new_practice_target(pool: list[dict]) -> dict:
    current    = st.session_state.get("practice_target")
    candidates = [c for c in pool if c != current] or pool
    return random.choice(candidates)


# ──────────────────────────────────────────────────────────────────────────────
# TAB LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
st.title(" Devanagari Character Recognizer")

if not st.session_state["model_loaded"]:
    st.warning(
        "**Model weights not found.** Place `best_model.pth` in the same folder "
        "as `app.py` and restart. The UI is fully functional but predictions will "
        "be random until weights are loaded.",
        icon="⚠️",
    )

tab_recog, tab_practice, tab_about = st.tabs(["✏️ Recognize", "🎯 Practice", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOGNIZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_recog:
    st.markdown(
        "Draw any Devanagari character in the box below. "
        "Predictions update when you click **Recognize**."
    )
    st.markdown("---")

    col_canvas, col_results = st.columns([1, 1], gap="large")

    with col_canvas:
        st.markdown("**Draw here** (white on black)")
        canvas_data = draw_canvas("recog", height=240)

        btn_col1, btn_col2 = st.columns(2)
        run_recog   = btn_col1.button("Recognize", type="primary", use_container_width=True)
        clear_recog = btn_col2.button("Clear",     use_container_width=True)

        if clear_recog:
            st.session_state["canvas_key_recog"] += 1
            st.session_state["last_result"] = None
            st.rerun()

    with col_results:
        st.markdown("**Results**")

        if run_recog:
            tensor = preprocess_canvas(canvas_data)
            if tensor is None:
                st.session_state["last_result"] = None
                st.warning("Nothing drawn yet — pick up your pen!")
            else:
                results = predict(tensor, top_k=3)
                st.session_state["last_result"] = results

        results = st.session_state["last_result"]

        if results is None:
            st.markdown(
                #"<div class='result-placeholder'>Draw a character and press <b>Recognize</b>.</div>" ,
                 "<div style='font-size:14px;padding-top:20px'>"
                 "Draw a character and press <b>Recognize</b>."
                 "</div>",
                unsafe_allow_html=True,
            )
        else:
            top_glyph, top_roman, top_conf = results[0]

            st.markdown(
                f"""
                <div class="glyph-card">
                  <div class="glyph-big">{top_glyph}</div>
                  <div class="glyph-label">/{top_roman}/ &nbsp;·&nbsp; {top_conf*100:.1f}% confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Top 3 predictions**")
            bar_colors = ["#4a7adf", "#7aab6e", "#c9954c"]
            for (g, r, c), color in zip(results, bar_colors):
                confidence_bar(f"{g}  /{r}/", c, color)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRACTICE MODE
# ══════════════════════════════════════════════════════════════════════════════
with tab_practice:
    st.markdown(
        "The app shows a target character. Draw it, then press **Check**. "
        "Practice until you can get 10 in a row!"
    )

    difficulty = st.radio(
        "Character pool",
        ["All 46", "Consonants only (36)", "Digits only (10)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if difficulty.startswith("Consonants"):
        pool = CONSONANTS
    elif difficulty.startswith("Digits"):
        pool = DIGITS
    else:
        pool = CLASSES_SORTED

    # Reset if pool changes
    if "last_pool_size" not in st.session_state:
        st.session_state["last_pool_size"] = len(pool)
    if len(pool) != st.session_state["last_pool_size"]:
        st.session_state["practice_target"]   = None
        st.session_state["practice_score"]    = 0
        st.session_state["practice_attempts"] = 0
        st.session_state["practice_streak"]   = 0
        st.session_state["practice_history"]  = []
        st.session_state["canvas_key_prac"]  += 1
        st.session_state["last_pool_size"]    = len(pool)

    if st.session_state["practice_target"] is None:
        st.session_state["practice_target"] = new_practice_target(pool)

    target = st.session_state["practice_target"]

    st.markdown("---")

    score    = st.session_state["practice_score"]
    attempts = st.session_state["practice_attempts"]
    streak   = st.session_state["practice_streak"]
    accuracy = (score / attempts * 100) if attempts > 0 else 0

    st.markdown(
        f"""
        <div class="score-row">
          <span class="score-pill pill-green">✓ {score} correct</span>
          <span class="score-pill pill-red">✗ {attempts - score} wrong</span>
          <span class="score-pill pill-blue">🔥 {streak} streak</span>
          <span class="score-pill pill-blue">📊 {accuracy:.0f}% accuracy</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if streak > 0 and streak % 5 == 0:
        st.success(f"🎉 {streak}-in-a-row streak! Keep it up!")

    st.markdown("---")

    col_target, col_draw, col_feedback = st.columns([1, 1, 1], gap="medium")

    with col_target:
        st.markdown("**Target character**")
        st.markdown(
            f"""
            <div class="target-card">
              <div class="target-glyph">{target['glyph']}</div>
              <div class="target-label">/{target['roman']}/</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:12px;text-align:center;margin-top:6px'>"
            "Copy this character in the drawing box</p>",
            unsafe_allow_html=True,
        )

    with col_draw:
        st.markdown("**Your drawing**")
        prac_canvas = draw_canvas("prac", height=200)

        check_btn = st.button("Check ✓", type="primary", use_container_width=True)
        skip_btn  = st.button("Skip →",  use_container_width=True)

    with col_feedback:
        st.markdown("**Feedback**")
        feedback_placeholder = st.empty()

        if check_btn:
            tensor = preprocess_canvas(prac_canvas)

            if tensor is None:
                feedback_placeholder.markdown(
                    "<div class='feedback-empty'>✏️ Draw the character first!</div>",
                    unsafe_allow_html=True,
                )
            else:
                results = predict(tensor, top_k=3)
                top_glyph, _, top_conf = results[0]
                top3_glyphs = [g for g, _, _ in results]
                in_top1 = (top_glyph == target["glyph"])
                in_top3 = (target["glyph"] in top3_glyphs)

                st.session_state["practice_attempts"] += 1

                if in_top1:
                    st.session_state["practice_score"]  += 1
                    st.session_state["practice_streak"] += 1
                    st.session_state["practice_history"].append(
                        {"target": target["glyph"], "pred": top_glyph, "correct": True}
                    )
                    feedback_placeholder.markdown(
                        f"<div class='feedback-correct'>"
                        f"✅ Correct! The model recognised <b>{top_glyph}</b> "
                        f"with {top_conf*100:.0f}% confidence."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.4)
                    st.session_state["practice_target"] = new_practice_target(pool)
                    st.session_state["canvas_key_prac"] += 1
                    st.rerun()

                elif in_top3:
                    rank = top3_glyphs.index(target["glyph"]) + 1
                    feedback_placeholder.markdown(
                        f"<div class='feedback-wrong'>"
                        f"🟡 Almost! Your character appeared as prediction #{rank} "
                        f"({target['glyph']} / {target['roman']}). "
                        f"Model's top guess was <b>{top_glyph}</b>. Try again!"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state["practice_streak"] = 0
                    st.session_state["practice_history"].append(
                        {"target": target["glyph"], "pred": top_glyph, "correct": False}
                    )

                else:
                    feedback_placeholder.markdown(
                        f"<div class='feedback-wrong'>"
                        f"❌ Not quite. Model predicted <b>{top_glyph}</b> "
                        f"({top_conf*100:.0f}%). "
                        f"Target was <b>{target['glyph']}</b> / {target['roman']}. "
                        f"Try again or skip."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state["practice_streak"] = 0
                    st.session_state["practice_history"].append(
                        {"target": target["glyph"], "pred": top_glyph, "correct": False}
                    )

        if skip_btn:
            st.session_state["practice_target"] = new_practice_target(pool)
            st.session_state["canvas_key_prac"] += 1
            st.session_state["practice_streak"] = 0
            st.rerun()

    # Practice history
    history = st.session_state["practice_history"]
    if len(history) >= 3:
        st.markdown("---")
        st.markdown("**Recent attempts** (latest first)")
        recent = list(reversed(history[-8:]))
        cols = st.columns(min(len(recent), 8))
        for col, entry in zip(cols, recent):
            icon  = "✅" if entry["correct"] else "❌"
            color = "#1a7a45" if entry["correct"] else "#9b2020"
            col.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:24px'>{entry['target']}</div>"
                f"<div style='font-size:11px;color:{color}'>{icon}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    if st.button("🔄 Reset scores"):
        st.session_state["practice_score"]    = 0
        st.session_state["practice_attempts"] = 0
        st.session_state["practice_streak"]   = 0
        st.session_state["practice_history"]  = []
        st.session_state["practice_target"]   = new_practice_target(pool)
        st.session_state["canvas_key_prac"]  += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    col1, col2 = st.columns(2)
    col1.metric("Classes",    "46")
    col1.metric("Parameters", "~300k")
    col2.metric("Training images", "~92k")
    col2.metric("Test accuracy",   "~97%")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
        **1. Drawing:** `streamlit-drawable-canvas` captures your strokes as RGBA pixels.

        **2. Preprocessing:**
        - RGB → grayscale → binary threshold (stroke vs background)
        - Cropped to bounding box of stroke
        - Resized to 28×28, padded to 32×32
        - Normalised to [−1, 1]

        **3. Model — CNN:**
        A 3-layer convolutional network trained from scratch on the
        [UCI Devanagari Handwritten Character Dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)
        (92,000 images, 46 classes).

        ```
        Input 1×32×32
          → Conv(32) + ReLU + MaxPool  → 32×16×16
          → Conv(64) + ReLU + MaxPool  → 64×8×8
          → Conv(128)+ ReLU + MaxPool  → 128×4×4
          → FC(256) + Dropout(0.2)
          → FC(46)  Softmax
        ```

        **4. Output:** Top-3 class probabilities shown as confidence bars.
        """
    )

    st.markdown("---")
    st.markdown("### Common confusions")
    st.markdown(
        """
        The model occasionally confuses visually similar pairs.
        Watch out for these when practicing:

        | Pair | Why they're similar |
        |------|---------------------|
        | ड / ढ | Same base shape; ढ adds an ascender stroke |
        | ब / व | Near-identical vertical structure |
        | श / ष | Mirror-like top strokes |
        | ण / न | Similar curved body |
        | ठ / ट | One extra horizontal stroke |
        """
    )
    
    st.markdown("---")
    st.markdown("### All 46 characters")

    st.markdown("**Consonants (36)**")
    consonant_html = "<div class='char-grid'>"
    for c in CONSONANTS:
        consonant_html += (
            f"<div class='char-cell'>{c['glyph']}"
            f"<span class='char-sub'>/{c['roman']}/</span></div>"
        )
    consonant_html += "</div>"
    st.markdown(consonant_html, unsafe_allow_html=True)

    st.markdown("<br>**Digits (10)**", unsafe_allow_html=True)
    digit_html = "<div class='char-grid'>"
    for c in DIGITS:
        digit_html += (
            f"<div class='char-cell'>{c['glyph']}"
            f"<span class='char-sub'>/{c['roman']}/</span></div>"
        )
    digit_html += "</div>"
    st.markdown(digit_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        ### References
        - Acharya, S. et al. (2015). *Deep learning based large scale handwritten Devanagari
          character recognition.* ICDAR.
        - [UCI ML Repository Dataset #389](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)
        - SMAI Assignment 3 — IIIT Hyderabad, 2025–26

    
        All evaluation and analysis by the student team.
        """
    )
