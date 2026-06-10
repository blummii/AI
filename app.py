import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import datetime

# ----------------- CONFIGURATION & STYLING -----------------
st.set_page_config(layout="wide", page_title="Banking NLP Dashboard")

# Inject exact match styling for dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #E2E8F0 !important;
    }
    
    .stApp {
        background-color: #121214;
    }
    
    /* Clean sidebar design in Dark Mode */
    section[data-testid="stSidebar"] {
        background-color: #1A1A1E !important;
        border-right: 1px solid #2C2C30;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
    }
    
    section[data-testid="stSidebar"] * {
        color: #A1A1AA !important;
    }
    
    /* Active list items in sidebar */
    div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        background-color: transparent !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        color: #A1A1AA !important;
        cursor: pointer !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #27272A !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #27272A !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stRadio"] label input {
        display: none !important;
    }
    
    /* Time duration selector as horizontal pills */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] label {
        padding: 6px 12px !important;
        border-radius: 16px !important;
        background-color: #27272A !important;
        border: 1px solid #3F3F46 !important;
        font-size: 13px !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
        color: #FFFFFF !important;
    }
    
    /* Style Streamlit's native bordered containers to look like premium dark cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1E1E22 !important;
        border: 1px solid #2C2C30 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 20px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Equal height columns styling */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] > div {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1 1 auto !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* Custom tab navigation styling in dark mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        padding: 4px 0;
        border-bottom: 1px solid #2C2C30;
        margin-bottom: 24px;
        border-radius: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #A1A1AA;
        background-color: transparent;
        padding: 10px 4px;
        font-weight: 700;
        font-size: 15px;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
        border-radius: 0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF;
    }
    
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    /* Make the default animated sliding tab line Indigo instead of red */
    div[data-baseweb="tab-highlight"] {
        background-color: #6366F1 !important;
    }
    
    /* Styled buttons */
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.2);
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        transform: translateY(-1px);
    }
    
    /* Checkbox styling */
    span[data-baseweb="checkbox"] {
        margin-bottom: 6px;
    }
    
    /* Checkbox active state styled as Indigo */
    [data-testid="stCheckbox"] input:checked + div span {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
    }
    input[type='checkbox'] {
        accent-color: #4F46E5 !important;
    }
    
    /* Slider thumb and track styled as Indigo */
    div[role="slider"] {
        background-color: #4F46E5 !important;
        border-color: #4F46E5 !important;
    }
    div[data-baseweb="slider"] > div > div {
        background-color: #4F46E5 !important;
    }
    
    /* Specific styling for pagination buttons to make them small, gray, with no blue hover */
    .st-key-prev_page_btn button, .st-key-next_page_btn button {
        background-color: #27272A !important;
        color: #E2E8F0 !important;
        border: 1px solid #3F3F46 !important;
        padding: 0 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.2s ease !important;
        transform: none !important;
        min-height: 0 !important;
        height: 32px !important;
        width: 36px !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 auto !important;
    }
    
    .st-key-prev_page_btn button *, .st-key-next_page_btn button * {
        color: #E2E8F0 !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        line-height: 1 !important;
    }
    
    .st-key-prev_page_btn button:hover, .st-key-next_page_btn button:hover {
        background-color: #3F3F46 !important;
        border-color: #52525B !important;
    }
    
    .st-key-prev_page_btn button:hover *, .st-key-next_page_btn button:hover * {
        color: #FFFFFF !important;
    }
    
    .st-key-prev_page_btn button:disabled, .st-key-next_page_btn button:disabled {
        background-color: #1A1A1E !important;
        border-color: #27272A !important;
        cursor: not-allowed !important;
    }
    
    .st-key-prev_page_btn button:disabled *, .st-key-next_page_btn button:disabled * {
        color: #52525B !important;
    }
    
    /* Refresh button styling: smaller, dark gray, right-aligned */
    .st-key-refresh_btn button {
        background-color: #27272A !important;
        color: #E2E8F0 !important;
        border: 1px solid #3F3F46 !important;
        padding: 4px 16px !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        height: 32px !important;
        width: auto !important;
        margin-left: auto !important;
        display: block !important;
        box-shadow: none !important;
        transform: none !important;
    }
    .st-key-refresh_btn button:hover {
        background-color: #3F3F46 !important;
        color: #FFFFFF !important;
        border-color: #52525B !important;
        transform: none !important;
    }
    
    /* Style st.info/stAlert alert boxes s.t. they match the dark card theme */
    div[data-testid="stAlert"] {
        background-color: #1E1E22 !important;
        border: 1px solid #2C2C30 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stAlert"] p {
        color: #E2E8F0 !important;
    }
    
    /* Style st.progress progress bars to be Indigo */
    div[data-testid="stProgress"] div[role="progressbar"] > div {
        background-color: #4F46E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED MODEL LOADING -----------------
@st.cache_resource
def load_nlp_model():
    model_path = "./best_viso_bert_balanced"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_nlp_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# Core Labels mapping (Khen/Chê)
LABEL_MAPPING = {
    0: "Tiêu cực",
    1: "Trung lập",
    2: "Tích cực"
}
LABEL_COLORS = {
    "Tiêu cực": "#FF6C6C",  # Vibrant Soft Red (matching the red pill)
    "Trung lập": "#FBBF24",  # Vibrant Amber Yellow (matching tone/brightness)
    "Tích cực": "#5CE488"   # Vibrant Minty Green (matching the green pill)
}

def render_html(html_str):
    return "\n".join(line.strip() for line in html_str.split("\n"))

# ----------------- SYNTHETIC DATA GENERATOR FOR DEFAULT STATE -----------------
def load_default_dataset():
    np.random.seed(42)
    n_rows = 5842
    
    # Banks list
    banks = ["MB Bank", "VCB Digibank", "BIDV Smart", "VPBank NEO", "Techcombank", "ACB ONE"]
    # Probabilities for each bank
    bank_probs = [0.18, 0.20, 0.15, 0.14, 0.18, 0.15]
    selected_banks = np.random.choice(banks, size=n_rows, p=bank_probs)
    
    sentiments = []
    csats = []
    aspects_list = []
    
    # Reviews pools
    pos_reviews = [
        "App ngân hàng dùng rất mượt mà, giao diện hiện đại bo viền đẹp mắt, chuyển tiền siêu nhanh.",
        "Chuyển tiền nhanh giao diện dễ dùng rất thích, màu sắc hài hòa đẹp đẽ.",
        "Sản phẩm dùng tốt, giao diện trực quan và giao dịch cực kì tiện lợi!",
        "FaceID nhạy, bảo mật cực tốt, tôi rất yên tâm sử dụng hàng ngày.",
        "Nhiều ưu đãi hoàn tiền, giao dịch hoàn toàn miễn phí chuyển khoản, rất tuyệt vời.",
        "Nhân viên tổng đài hỗ trợ nhiệt tình, tư vấn chu đáo qua hotline."
    ]
    neu_reviews = [
        "Mới cập nhật ứng dụng bản mới ngày hôm qua, chưa thấy đổi mới gì nhiều.",
        "Cho tôi hỏi cách đổi mật khẩu ngân hàng số như thế nào trên giao diện mới?",
        "Ứng dụng bình thường, không có gì quá nổi bật so với các app khác.",
        "Đang tìm hiểu các tính năng mới của app xem có lỗi lầm gì không.",
        "Cần thêm thời gian để đánh giá chất lượng dịch vụ vì tôi mới dùng app."
    ]
    neg_reviews = [
        "Ứng dụng hay bị lag, đăng nhập mãi không được, nhiều lúc tự thoát ra ngoài rất bực mình. Giao diện thì khó dùng.",
        "Phí dịch vụ chuyển tiền của ngân hàng này hơi cao so với các bên khác.",
        "FaceID không hoạt động được sau khi nâng cấp hệ điều hành mới nhất, app bị crash hoài.",
        "Phản hồi chậm chạp lỗi chuyển khoản hoài bực mình ghê, cập nhật xong tệ hẳn.",
        "Giao diện mới nhìn rất rối mắt và khó sử dụng hơn bản cũ, chữ nhỏ khó nhìn.",
        "Mỗi lần cập nhật xong lại đơ lag ứng dụng, phí dịch vụ thì đắt đỏ quá."
    ]
    
    # Aspect labels
    aspect_mapping_labels = ["Bảo mật & Đăng nhập", "Giao dịch & Chuyển tiền", "Giao diện & Trải nghiệm", "Phí dịch vụ & Khuyến mãi", "Hỗ trợ khách hàng"]
    
    for bank in selected_banks:
        if bank == "MB Bank":
            p_pos, p_neu, p_neg = 0.68, 0.12, 0.20
        elif bank == "VCB Digibank":
            p_pos, p_neu, p_neg = 0.71, 0.15, 0.14
        elif bank == "BIDV Smart":
            p_pos, p_neu, p_neg = 0.56, 0.14, 0.30
        elif bank == "VPBank NEO":
            p_pos, p_neu, p_neg = 0.60, 0.16, 0.24
        elif bank == "Techcombank":
            p_pos, p_neu, p_neg = 0.74, 0.12, 0.14
        else: # ACB ONE
            p_pos, p_neu, p_neg = 0.58, 0.18, 0.24
            
        sent = np.random.choice(["Tích cực", "Trung lập", "Tiêu cực"], p=[p_pos, p_neu, p_neg])
        sentiments.append(sent)
        
        if sent == "Tích cực":
            csats.append(round(np.random.uniform(4.0, 5.0), 2))
            aspects_list.append(np.random.choice(aspect_mapping_labels, p=[0.2, 0.3, 0.3, 0.1, 0.1]))
        elif sent == "Trung lập":
            csats.append(round(np.random.uniform(2.5, 3.5), 2))
            aspects_list.append(np.random.choice(["Giao diện & Trải nghiệm", "Giao dịch & Chuyển tiền", "Khác"], p=[0.4, 0.4, 0.2]))
        else:
            csats.append(round(np.random.uniform(1.0, 2.0), 2))
            aspects_list.append(np.random.choice(aspect_mapping_labels, p=[0.3, 0.3, 0.3, 0.1, 0.0]))
            
    # Generate random dates between 2026-01-01 and 2026-06-06
    start_date = datetime.datetime(2026, 1, 1)
    end_date = datetime.datetime(2026, 6, 6)
    delta = end_date - start_date
    random_days = np.random.randint(0, delta.days + 1, size=n_rows)
    dates = [start_date + datetime.timedelta(days=int(d)) for d in random_days]
    
    # Generate texts
    texts = []
    for s in sentiments:
        if s == "Tích cực":
            texts.append(pos_reviews[np.random.randint(0, len(pos_reviews))])
        elif s == "Trung lập":
            texts.append(neu_reviews[np.random.randint(0, len(neu_reviews))])
        else:
            texts.append(neg_reviews[np.random.randint(0, len(neg_reviews))])
            
    df_large = pd.DataFrame({
        "Ngân hàng": selected_banks,
        "Ngày đánh giá": dates,
        "Nội dung phản hồi": texts,
        "CSAT_Score": csats,
        "Sentiment": sentiments,
        "Aspects": [[a] for a in aspects_list]
    })
    
    label_map_rev = {"Tiêu cực": 0, "Trung lập": 1, "Tích cực": 2}
    df_large["Nhãn cảm xúc"] = df_large["Sentiment"].map(label_map_rev)
    df_large["Probabilities"] = df_large["Sentiment"].apply(lambda s: [0.87, 0.09, 0.04] if s == "Tiêu cực" else [0.08, 0.81, 0.11] if s == "Trung lập" else [0.03, 0.08, 0.89])
    
    return df_large

# Aspect ABSA simulation using keywords
def extract_aspects(text):
    text_lower = str(text).lower()
    aspects = []
    if any(w in text_lower for w in ["đăng nhập", "login", "faceid", "vân tay", "face id", "mật khẩu", "otp", "bảo mật", "security", "khóa"]):
        aspects.append("Bảo mật & Đăng nhập")
    if any(w in text_lower for w in ["chuyển tiền", "chuyển khoản", "ck", "giao dịch", "nap tien", "nạp tiền", "rút tiền", "banking", "tài khoản", "số dư", "hạn mức"]):
        aspects.append("Giao dịch & Chuyển tiền")
    if any(w in text_lower for w in ["giao diện", "ui", "ux", "màn hình", "font", "chữ", "đẹp", "xấu", "màu", "thiết kế", "dễ dùng", "khó dùng", "lag", "đơ", "chậm", "văng", "crash"]):
        aspects.append("Giao diện & Trải nghiệm")
    if any(w in text_lower for w in ["phí", "free", "miễn phí", "hoàn tiền", "khuyến mãi", "ưu đãi", "quà"]):
        aspects.append("Phí dịch vụ & Khuyến mãi")
    if any(w in text_lower for w in ["hỗ trợ", "hotline", "nhân viên", "tổng đài", "chatbot", "chat", "phục vụ", "tư vấn"]):
        aspects.append("Hỗ trợ khách hàng")
    
    if not aspects:
        aspects.append("Khác")
    return aspects

def run_model_inference(df, text_column, batch_size=16):
    df_result = df.copy()
    predictions = []
    probabilities = []
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    total_rows = len(df_result)
    num_batches = int(np.ceil(total_rows / batch_size))
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        
        # Ép kiểu dữ liệu từng dòng nhận xét sang chuỗi ký tự (string), xử lý giá trị khuyết (NaN/None) thành chuỗi rỗng
        # Sử dụng List Comprehension để đảm bảo an toàn tuyệt đối trên Python 3.13 (tránh lỗi crash của tokenizer)
        batch_raw = df_result[text_column].iloc[start_idx:end_idx].tolist()
        batch_texts = [str(t) if pd.notna(t) else "" for t in batch_raw]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_probs = torch.softmax(outputs.logits, dim=-1)
            
            batch_preds = torch.argmax(batch_probs, dim=-1).cpu().numpy()
            batch_probs_np = batch_probs.cpu().numpy()
            
            predictions.extend([LABEL_MAPPING[p] for p in batch_preds])
            probabilities.extend(batch_probs_np.tolist())
            
        progress_percent = float(end_idx / total_rows)
        progress_bar.progress(progress_percent)
        status_text.text(f"Đang phân tích: {end_idx}/{total_rows}...")
        
    progress_bar.empty()
    status_text.empty()
    
    df_result['Sentiment'] = predictions
    df_result['Probabilities'] = probabilities
    
    csat_scores = []
    for probs in probabilities:
        csat = (probs[0] * 1) + (probs[1] * 3) + (probs[2] * 5)
        csat_scores.append(round(csat, 2))
    df_result['CSAT_Score'] = csat_scores
    df_result['Aspects'] = df_result[text_column].apply(extract_aspects)
    
    return df_result

# ----------------- REAL LIME EXPLAINER -----------------
def explain_text_lime(text, tokenizer, model):
    words = text.split()
    if not words:
        return [], [], 1
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        base_probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_class = torch.argmax(base_probs).item()
        base_prob = base_probs[pred_class].item()
        
    importances = []
    for i in range(len(words)):
        perturbed_words = words[:i] + words[i+1:]
        perturbed_text = " ".join(perturbed_words)
        if not perturbed_text.strip():
            importances.append(0.0)
            continue
            
        inputs_p = tokenizer(perturbed_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs_p = model(**inputs_p)
            p_probs = torch.softmax(outputs_p.logits, dim=-1)[0]
            p_prob = p_probs[pred_class].item()
        
        importances.append(base_prob - p_prob)
        
    return words, importances, pred_class

# ----------------- CONFUSION MATRIX HTML RENDERER -----------------
def render_html_confusion_matrix(cm):
    html = f"""
    <div style="font-family: 'Plus Jakarta Sans'; padding: 10px; background-color: transparent;">
        <div style="text-align: center; color: #71717A; font-size: 12px; margin-bottom: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Nhãn dự đoán ➔</div>
        <div style="display: grid; grid-template-columns: 100px repeat(3, 1fr); gap: 10px; text-align: center; align-items: center;">
            <div></div>
            <div style="color: #A1A1AA; font-weight: 700; font-size: 13px;">Tiêu cực</div>
            <div style="color: #A1A1AA; font-weight: 700; font-size: 13px;">Trung tính</div>
            <div style="color: #A1A1AA; font-weight: 700; font-size: 13px;">Tích cực</div>
            
            <div style="color: #A1A1AA; font-weight: 700; text-align: right; padding-right: 12px; font-size: 13px;">Tiêu cực</div>
            <div style="background-color: #412426; color: #FF6C6C; border: 1px solid rgba(255, 108, 108, 0.3); font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[0][0]}</div>
            <div style="background-color: #3F3F46; color: #FFFFFF; font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[0][1]}</div>
            <div style="background-color: #27272A; color: #A1A1AA; font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[0][2]}</div>
            
            <div style="color: #A1A1AA; font-weight: 700; text-align: right; padding-right: 12px; font-size: 13px;">Trung tính</div>
            <div style="background-color: #3F3F46; color: #FFFFFF; font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[1][0]}</div>
            <div style="background-color: #4F3C16; color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.3); font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[1][1]}</div>
            <div style="background-color: #27272A; color: #A1A1AA; font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[1][2]}</div>
            
            <div style="color: #A1A1AA; font-weight: 700; text-align: right; padding-right: 12px; font-size: 13px;">Tích cực</div>
            <div style="background-color: #27272A; color: #A1A1AA; font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[2][0]}</div>
            <div style="background-color: #3F3F46; color: #FFFFFF; font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[2][1]}</div>
            <div style="background-color: #1a3926; color: #5CE488; border: 1px solid rgba(92, 228, 136, 0.3); font-weight: 800; border-radius: 8px; padding: 18px 0; font-size: 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">{cm[2][2]}</div>
        </div>
        <div style="margin-top: 14px; font-size: 11px; color: #71717A; text-align: center; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">← Nhãn thực tế</div>
    </div>
    """
    return render_html(html)

def extract_top_keywords(df, sentiment, top_n=8):
    subset = df[df['Sentiment'] == sentiment]
    if subset.empty:
        return []
    
    import re
    from collections import Counter
    
    phrases = [
        "chuyển tiền", "đăng nhập", "giao diện", "dễ dùng", "khó dùng", "văng app", 
        "face id", "phí dịch vụ", "nạp tiền", "rút tiền", "khuyến mãi", "ưu đãi",
        "tổng đài", "nhân viên", "mượt mà", "nhanh chóng", "chậm chạp", "lỗi mạng", "bảo mật"
    ]
    
    stopwords = {
        "và", "của", "là", "cho", "để", "các", "như", "có", "cũng", "được", "bị", "với", "này", 
        "app", "ngân", "hàng", "ứng", "dụng", "tôi", "anh", "chị", "em", "nha", "nhé", "nè", 
        "quá", "rất", "hơn", "nhiều", "lại", "đi", "vào", "ra", "lên", "xuống", "cả", "chỉ", 
        "thì", "mà", "nhưng", "nào", "gì", "sao", "thế", "vậy", "đang", "đã", "sẽ", "được",
        "chưa", "không", "một", "hai", "ba", "cái", "này", "còn", "nên", "cho", "như", "những"
    }
    
    counter = Counter()
    # Nhận diện động cột chứa nội dung nhận xét của khách hàng từ session state
    # Nếu không tìm thấy cột hợp lệ, tự động fallback lấy cột kiểu chữ (object) đầu tiên trong file CSV
    txt_col = st.session_state.get('detected_txt_col')
    if not txt_col or txt_col not in subset.columns:
        txt_col = subset.select_dtypes(include=['object']).columns[0] if len(subset.select_dtypes(include=['object']).columns) > 0 else subset.columns[0]
    
    # Ép kiểu dữ liệu sang chuỗi ký tự (str) để tránh lỗi 'float has no attribute lower' trên Python 3.13 ở local
    raw_texts = subset[txt_col].tolist()
    safe_texts = [str(t) if pd.notna(t) else "" for t in raw_texts]
    for text in safe_texts:
        text_lower = text.lower()
        temp_text = text_lower
        for phrase in phrases:
            matches = len(re.findall(r'\b' + re.escape(phrase) + r'\b', temp_text))
            if matches > 0:
                counter[phrase] += matches
                temp_text = re.sub(r'\b' + re.escape(phrase) + r'\b', ' ', temp_text)
        
        words = re.findall(r'\b\w+\b', temp_text)
        for w in words:
            if len(w) > 2 or w in ["lỗi", "phí", "lag", "đơ", "ok", "tệ", "tốt", "đẹp", "rối", "chậm", "nhanh"]:
                if w not in stopwords:
                    counter[w] += 1
                    
    return [item[0] for item in counter.most_common(top_n)]

# ----------------- SESSION STATE INITIALIZATION -----------------
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
    st.session_state['classified_data'] = None
    st.session_state['file_name'] = None
    st.session_state['detected_txt_col'] = None
    st.session_state['detected_app_col'] = None
    st.session_state['detected_date_col'] = None
    st.session_state['detected_label_col'] = None
    st.session_state['is_demo_data'] = False


if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 1

# ----------------- INFERENCE ON UPLOADED DATA -----------------
if st.session_state['raw_data'] is not None and st.session_state['classified_data'] is None:
    df = st.session_state['raw_data']
    
    if st.session_state.get('is_demo_data', False):
        # Bypass inference for the demo/simulated dataset (it already has pre-computed sentiment labels & aspects)
        st.session_state['classified_data'] = df
        st.session_state['detected_txt_col'] = "Nội dung phản hồi"
        st.session_state['detected_app_col'] = "Ngân hàng"
        st.session_state['detected_date_col'] = "Ngày đánh giá"
        st.session_state['detected_label_col'] = "Nhãn cảm xúc"
    else:
        cols = list(df.columns)
        
        # 1. TỰ ĐỘNG NHẬN DIỆN CỘT DỮ LIỆU (Chỉ chạy lần đầu tiên khi mới tải file CSV/Excel lên)
        if st.session_state.get('detected_txt_col') not in cols:
            # Quét tìm các cột dựa trên các từ khóa thông dụng tiếng Việt và tiếng Anh
            txt_col = next((c for c in cols if any(k in c.lower() for k in ["text", "review", "binh luan", "content", "noi dung", "nội dung", "feedback", "phan hoi", "phản hồi", "đánh giá", "danh gia"]) and not any(dk in c.lower() for dk in ["ngày", "ngay", "date", "time"])), cols[0])
            app_col = next((c for c in cols if any(k in c.lower() for k in ["app", "bank", "ngan hang", "ngân hàng", "ten_app", "application"])), None)
            date_col = next((c for c in cols if any(k in c.lower() for k in ["date", "time", "ngay", "ngày", "timestamp", "thời gian", "thoi gian"])), None)
            label_col = next((c for c in cols if any(k in c.lower() for k in ["label", "sentiment", "nhan", "nhãn", "y_true", "ground_truth", "cam xuc", "cảm xúc"])), None)
            
            st.session_state['detected_txt_col'] = txt_col
            st.session_state['detected_app_col'] = app_col
            st.session_state['detected_date_col'] = date_col
            st.session_state['detected_label_col'] = label_col
        else:
            # Nếu người dùng chọn thủ công trên giao diện chính, lấy cột tương ứng trực tiếp từ session_state
            txt_col = st.session_state['detected_txt_col']
            app_col = st.session_state['detected_app_col']
            date_col = st.session_state['detected_date_col']
            label_col = st.session_state['detected_label_col']
            
        # 2. KIỂM TRA TÍNH HỢP LỆ CỦA CỘT NHẬN XÉT (Tránh lỗi phân tích trên cột số, ngày tháng, hoặc cột trống)
        is_invalid = False
        if txt_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[txt_col]) or pd.api.types.is_datetime64_any_dtype(df[txt_col]) or df[txt_col].isnull().all():
                is_invalid = True
                
        if is_invalid:
            # Gán thông báo lỗi và đặt classified_data về None để chặn không chạy suy luận vô nghĩa
            st.session_state['data_error'] = f"Cột \"{txt_col}\" không chứa văn bản đánh giá hợp lệ. Vui lòng chọn lại cột chứa nhận xét của khách hàng để bắt đầu phân tích."
            st.session_state['classified_data'] = None
        else:
            st.session_state['data_error'] = None
            # 3. CHẠY SUY LUẬN MÔ HÌNH HỌC SÂU (Chỉ chạy khi dữ liệu chưa được xử lý cảm xúc)
            if st.session_state.get('classified_data') is None:
                if model_loaded:
                    classified_df = run_model_inference(df, txt_col)
                    st.session_state['classified_data'] = classified_df


# ----------------- SIDEBAR DATA SOURCE (ALWAYS RENDERED) -----------------
st.sidebar.markdown("<p style='font-size: 11px; font-weight: 700; color: #8E8E93; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em;'>NGUỒN DỮ LIỆU</p>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Tải file dữ liệu", type=["csv", "xlsx"], label_visibility="collapsed")

# Checkbox to use simulated/demo data
use_demo = st.sidebar.checkbox(
    "Sử dụng dữ liệu giả lập (Demo)",
    value=st.session_state.get('is_demo_data', False),
    disabled=(uploaded_file is not None),
    help="Chỉ khả dụng khi chưa tải lên tệp dữ liệu riêng."
)

# Reactive logic to handle uploaded file vs demo checkbox
if uploaded_file is not None:
    if st.session_state.get('file_name', '') != uploaded_file.name:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state['raw_data'] = df
            st.session_state['file_name'] = uploaded_file.name
            st.session_state['classified_data'] = None
            st.session_state['is_demo_data'] = False
            # Reset detected columns so they auto-detect on the new file
            st.session_state['detected_txt_col'] = None
            st.session_state['detected_app_col'] = None
            st.session_state['detected_date_col'] = None
            st.session_state['detected_label_col'] = None
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Lỗi mở file: {e}")
elif use_demo:
    if not st.session_state.get('is_demo_data', False) or st.session_state['raw_data'] is None:
        try:
            df = load_default_dataset()
            st.session_state['raw_data'] = df
            st.session_state['file_name'] = "Simulated_Data_Demo"
            st.session_state['classified_data'] = None
            st.session_state['is_demo_data'] = True
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Lỗi tạo dữ liệu giả lập: {e}")
else:
    # If no file uploaded and use_demo checkbox is false, but we currently have demo data loaded, clear it
    if st.session_state.get('is_demo_data', False) or st.session_state['raw_data'] is not None:
        st.session_state['raw_data'] = None
        st.session_state['file_name'] = None
        st.session_state['classified_data'] = None
        st.session_state['is_demo_data'] = False
        st.rerun()

# Column configuration has been moved to the main area header for improved UX.

c_df = st.session_state['classified_data']

if st.session_state['raw_data'] is not None and c_df is None:
    st.info("Đang xử lý dữ liệu và chạy mô hình phân tích... Vui lòng đợi trong giây lát.")
    st.stop()

if c_df is not None:
    txt_col = st.session_state['detected_txt_col']
    app_col = st.session_state['detected_app_col']
    date_col = st.session_state['detected_date_col']
    label_col = st.session_state['detected_label_col']
    
    # ----------------- SIDEBAR CONTROLS -----------------
    # 1. ỨNG DỤNG list radio selector
    st.sidebar.markdown("<p style='font-size: 11px; font-weight: 700; color: #8E8E93; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em;'>ỨNG DỤNG</p>", unsafe_allow_html=True)
    
    if app_col and app_col in c_df.columns:
        unique_banks = ["Tất cả"] + sorted(c_df[app_col].dropna().unique().tolist())
    else:
        unique_banks = ["Tất cả"]
    
    selected_app_name = st.sidebar.radio("ỨNG DỤNG", unique_banks, index=0, label_visibility="collapsed")
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # 2. THỜI GIAN date range slider selector
    st.sidebar.markdown("<p style='font-size: 11px; font-weight: 700; color: #8E8E93; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em;'>KHOẢNG THỜI GIAN</p>", unsafe_allow_html=True)
    selected_dates = None
    if date_col and date_col in c_df.columns:
        c_df[date_col] = pd.to_datetime(c_df[date_col], errors='coerce')
        valid_dates = c_df[date_col].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            if min_date == max_date:
                st.sidebar.info(f"Thời gian: {min_date.strftime('%d/%m/%Y')}")
                selected_dates = (min_date, max_date)
            else:
                selected_dates = st.sidebar.slider(
                    "Khoảng thời gian",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date),
                    format="DD/MM/YYYY",
                    label_visibility="collapsed"
                )
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # 3. NHÃN Checkboxes
    st.sidebar.markdown("<p style='font-size: 11px; font-weight: 700; color: #8E8E93; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em;'>NHÃN CẢM XÚC</p>", unsafe_allow_html=True)
    show_pos = st.sidebar.checkbox("Tích cực", value=True)
    show_neu = st.sidebar.checkbox("Trung tính", value=True)
    show_neg = st.sidebar.checkbox("Tiêu cực", value=True)
    
    selected_sentiments = []
    if show_pos: selected_sentiments.append("Tích cực")
    if show_neu: selected_sentiments.append("Trung lập")
    if show_neg: selected_sentiments.append("Tiêu cực")
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # 4. OLAP CONTROLS
    st.sidebar.markdown("<p style='font-size: 11px; font-weight: 700; color: #8E8E93; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em;'>TÙY CHỌN BIỂU ĐỒ</p>", unsafe_allow_html=True)
    
    analysis_attrib = st.sidebar.selectbox(
        "Số liệu hiển thị",
        options=["Số lượng đánh giá", "Điểm CSAT trung bình", "Chỉ số NSS"],
        index=0
    )
    
    breakdown_options = []
    if app_col and app_col in c_df.columns:
        breakdown_options.append("Ngân hàng")
    breakdown_options.extend(["Khía cạnh / Tính năng", "Cảm xúc"])
    
    breakdown_dim = st.sidebar.selectbox(
        "Phân nhóm theo",
        options=breakdown_options,
        index=0
    )
    
    # ----------------- FILTER DATA -----------------
    filtered_df = c_df.copy()
    
    # Filter by selected app
    if selected_app_name != "Tất cả" and app_col and app_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[app_col] == selected_app_name]
    
    # Filter by date range slider selection
    if date_col and date_col in filtered_df.columns and selected_dates is not None:
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
        filtered_df = filtered_df[(filtered_df[date_col].dt.date >= selected_dates[0]) & (filtered_df[date_col].dt.date <= selected_dates[1])]
    
    # Filter by selected sentiments
    filtered_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiments)]
    
    # Map selected breakdown dimension to actual column name
    if breakdown_dim == "Ngân hàng":
        selected_dim_col = app_col if (app_col and app_col in filtered_df.columns) else None
    elif breakdown_dim == "Khía cạnh / Tính năng":
        selected_dim_col = "Aspects"
    else: # Cảm xúc
        selected_dim_col = "Sentiment"
    
    # Fallback check
    if selected_dim_col is None or (selected_dim_col not in filtered_df.columns and selected_dim_col != "Aspects"):
        selected_dim_col = "Sentiment"
        breakdown_dim_label = "Cảm xúc"
    else:
        breakdown_dim_label = breakdown_dim
else:
    # Set default values for variables to avoid NameErrors in the script
    filtered_df = pd.DataFrame()
    breakdown_dim_label = ""
    selected_dim_col = ""
    analysis_attrib = ""
    breakdown_dim = ""
    txt_col = ""
    app_col = ""
    date_col = ""
    label_col = ""
    st.sidebar.info("📂 Vui lòng tải lên file dữ liệu (.csv, .xlsx) hoặc chọn 'Sử dụng dữ liệu giả lập (Demo)' để kích hoạt các bộ lọc phân tích.")

# Premium harmonious color palette for categories (non-sentiments) - Modern Fintech Indigo
PREMIUM_PALETTE = [
    "#818CF8",  # Sắc độ 1 (Sáng nhất)
    "#6366F1",  # Sắc độ 2
    "#4F46E5",  # Sắc độ 3
    "#4338CA",  # Sắc độ 4
    "#3730A3",  # Sắc độ 5
    "#312E81"   # Sắc độ 6 (Tối nhất)
]

# Color configuration for plotly charts to avoid TypeError
color_args = {}
if selected_dim_col == 'Sentiment':
    color_args['color_discrete_map'] = LABEL_COLORS
else:
    color_args['color_discrete_sequence'] = PREMIUM_PALETTE

# ----------------- TOP BAR HEADER -----------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Banking NLP Dashboard")
with col_h2:
    st.write("") # Spacing
    if st.button("Làm mới", key="refresh_btn", width="stretch"):
        st.rerun()

# ----------------- MAIN AREA COLUMN CONFIGURATION (FOR CUSTOM UPLOADS) -----------------
# PHÂN HỆ KHAI BÁO CỘT PHÂN TÍCH (Chỉ hiển thị cho file dữ liệu tự tải lên)
# Thiết kế dưới dạng expander thu gọn mặc định để tối ưu hóa không gian màn hình chính
if st.session_state.get('raw_data') is not None and not st.session_state.get('is_demo_data', False):
    with st.expander("Chọn cột phân tích", expanded=False):
        df_temp = st.session_state['raw_data']
        cols = list(df_temp.columns)
        
        # Initialize and validate detected columns
        curr_txt = st.session_state.get('detected_txt_col')
        if curr_txt not in cols:
            curr_txt = next((c for c in cols if any(k in c.lower() for k in ["text", "review", "binh luan", "content", "noi dung", "nội dung", "feedback", "phan hoi", "phản hồi", "đánh giá", "danh gia"]) and not any(dk in c.lower() for dk in ["ngày", "ngay", "date", "time"])), cols[0])
            st.session_state['detected_txt_col'] = curr_txt
            
        curr_app = st.session_state.get('detected_app_col')
        if curr_app not in cols and curr_app is not None:
            curr_app = next((c for c in cols if any(k in c.lower() for k in ["app", "bank", "ngan hang", "ngân hàng", "ten_app", "application"])), None)
            st.session_state['detected_app_col'] = curr_app
            
        curr_date = st.session_state.get('detected_date_col')
        if curr_date not in cols and curr_date is not None:
            curr_date = next((c for c in cols if any(k in c.lower() for k in ["date", "time", "ngay", "ngày", "timestamp", "thời gian", "thoi gian"])), None)
            st.session_state['detected_date_col'] = curr_date
            
        curr_lbl = st.session_state.get('detected_label_col')
        if curr_lbl not in cols and curr_lbl is not None:
            curr_lbl = next((c for c in cols if any(k in c.lower() for k in ["label", "sentiment", "nhan", "nhãn", "y_true", "ground_truth", "cam xuc", "cảm xúc"])), None)
            st.session_state['detected_label_col'] = curr_lbl

        # Layout columns inside expander (2x2 grid for compact clean layout)
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            txt_idx = cols.index(st.session_state['detected_txt_col']) if st.session_state['detected_txt_col'] in cols else 0
            sel_txt = st.selectbox("Nội dung đánh giá", cols, index=txt_idx, key="sel_txt_col")
            
            date_options = ["Không có"] + cols
            date_val = st.session_state['detected_date_col'] if st.session_state['detected_date_col'] is not None else "Không có"
            date_idx = date_options.index(date_val) if date_val in date_options else 0
            sel_date = st.selectbox("Ngày đánh giá", date_options, index=date_idx, key="sel_date_col")
            
        with col_cfg2:
            app_options = ["Không có"] + cols
            app_val = st.session_state['detected_app_col'] if st.session_state['detected_app_col'] is not None else "Không có"
            app_idx = app_options.index(app_val) if app_val in app_options else 0
            sel_app = st.selectbox("Tên ngân hàng", app_options, index=app_idx, key="sel_app_col")
            
            lbl_options = ["Không có"] + cols
            lbl_val = st.session_state['detected_label_col'] if st.session_state['detected_label_col'] is not None else "Không có"
            lbl_idx = lbl_options.index(lbl_val) if lbl_val in lbl_options else 0
            sel_lbl = st.selectbox("Nhãn cảm xúc gốc", lbl_options, index=lbl_idx, key="sel_lbl_col")

        # Handle value changes
        new_app = sel_app if sel_app != "Không có" else None
        new_date = sel_date if sel_date != "Không có" else None
        new_lbl = sel_lbl if sel_lbl != "Không có" else None
        
        if (sel_txt != st.session_state.get('detected_txt_col') or 
            new_app != st.session_state.get('detected_app_col') or 
            new_date != st.session_state.get('detected_date_col') or 
            new_lbl != st.session_state.get('detected_label_col')):
            
            st.session_state['detected_txt_col'] = sel_txt
            st.session_state['detected_app_col'] = new_app
            st.session_state['detected_date_col'] = new_date
            st.session_state['detected_label_col'] = new_lbl
            
            st.session_state['classified_data'] = None
            st.rerun()

    # Hiển thị thông báo lỗi nếu cột văn bản nhận xét không chứa dữ liệu hợp lệ
    if st.session_state.get('data_error') is not None:
        st.error(st.session_state['data_error'])
        st.stop()

# ----------------- TABS CREATION -----------------
tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan", "Phân tích sâu", "Demo Live", "Giới thiệu"])

# ----------------- TAB 1: TỔNG QUAN -----------------
with tab1:
    if c_df is None:
        st.markdown("""
        <div style='background-color: #1E1E22; border: 1px solid #2C2C30; border-radius: 12px; padding: 40px; text-align: center; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); margin-top: 20px;'>
            <h3 style='color: #FFFFFF; font-weight: 700; margin-bottom: 15px; font-family: "Plus Jakarta Sans";'>Chưa có dữ liệu phân tích</h3>
            <p style='color: #A1A1AA; font-size: 15px; line-height: 1.6; max-width: 600px; margin: 0 auto 25px auto; font-family: "Plus Jakarta Sans";'>
                Hệ thống phân tích cảm xúc khách hàng đối với các ứng dụng ngân hàng số sử dụng mô hình học sâu học máy tiếng Việt <b>ViSoBERT</b>.
            </p>
            <div style='display: inline-block; background-color: rgba(79, 70, 229, 0.05); border: 1px dashed #3F3F46; border-radius: 8px; padding: 15px 25px; color: #A1A1AA; font-weight: 600; font-size: 14px; font-family: "Plus Jakarta Sans";'>
                📂 Vui lòng tải lên file dữ liệu (.csv, .xlsx) ở thanh bên trái hoặc chọn "Sử dụng dữ liệu giả lập (Demo)" để bắt đầu!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if filtered_df.empty:
            st.warning("Không tìm thấy kết quả thỏa mãn bộ lọc hiện tại.")
        else:
            # KPI calculations
            total_reviews = len(filtered_df)
            sent_counts = filtered_df['Sentiment'].value_counts()
            
            pos_count = sent_counts.get('Tích cực', 0)
            neg_count = sent_counts.get('Tiêu cực', 0)
            neu_count = sent_counts.get('Trung lập', 0)
            
            pos_pct = (pos_count / total_reviews * 100) if total_reviews > 0 else 0
            neg_pct = (neg_count / total_reviews * 100) if total_reviews > 0 else 0
            neu_pct = (neu_count / total_reviews * 100) if total_reviews > 0 else 0
            
            # Calculate NSS and Average CSAT
            nss_score = pos_pct - neg_pct
            avg_csat = filtered_df['CSAT_Score'].mean() if total_reviews > 0 else 0.0
            
            # Render KPI Row (4 Columns) using custom HTML to bypass Streamlit's default metric arrows and unify styling
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            pill_style = "display: inline-block; font-size: 12px; color: #A1A1AA; background-color: rgba(161, 161, 170, 0.15); border: 1px solid rgba(161, 161, 170, 0.2); border-radius: 16px; padding: 4px 12px; font-weight: 600;"
            
            with kpi_col1:
                st.markdown(f"""
                <div style='background-color: #1E1E22; border: 1px solid #2C2C30; border-radius: 12px; padding: 20px 24px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); width: 100%; box-sizing: border-box; height: 150px; display: flex; flex-direction: column; justify-content: space-between; font-family: "Plus Jakarta Sans";'>
                    <div style='font-size: 14px; color: #A1A1AA; font-weight: 500;'>Tổng reviews</div>
                    <div style='font-size: 32px; color: #FFFFFF; font-weight: 700; line-height: 1.1;'>{total_reviews:,}</div>
                    <div>
                        <div style='{pill_style}'>Dữ liệu đã chọn</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col2:
                st.markdown(f"""
                <div style='background-color: #1E1E22; border: 1px solid #2C2C30; border-radius: 12px; padding: 20px 24px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); width: 100%; box-sizing: border-box; height: 150px; display: flex; flex-direction: column; justify-content: space-between; font-family: "Plus Jakarta Sans";'>
                    <div style='font-size: 14px; color: #A1A1AA; font-weight: 500;'>Chỉ số cảm xúc ròng (NSS)</div>
                    <div style='font-size: 32px; color: #FFFFFF; font-weight: 700; line-height: 1.1;'>{nss_score:+.1f}%</div>
                    <div>
                        <div style='{pill_style}'>{pos_pct:.0f}% Tích cực | {neg_pct:.0f}% Tiêu cực</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col3:
                st.markdown(f"""
                <div style='background-color: #1E1E22; border: 1px solid #2C2C30; border-radius: 12px; padding: 20px 24px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); width: 100%; box-sizing: border-box; height: 150px; display: flex; flex-direction: column; justify-content: space-between; font-family: "Plus Jakarta Sans";'>
                    <div style='font-size: 14px; color: #A1A1AA; font-weight: 500;'>Điểm hài lòng CSAT</div>
                    <div style='font-size: 32px; color: #FFFFFF; font-weight: 700; line-height: 1.1;'>{avg_csat:.2f} / 5.0</div>
                    <div>
                        <div style='{pill_style}'>Dựa trên {total_reviews:,} đánh giá</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col4:
                st.markdown(f"""
                <div style='background-color: #1E1E22; border: 1px solid #2C2C30; border-radius: 12px; padding: 20px 24px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); width: 100%; box-sizing: border-box; height: 150px; display: flex; flex-direction: column; justify-content: space-between; font-family: "Plus Jakarta Sans";'>
                    <div style='font-size: 14px; color: #A1A1AA; font-weight: 500;'>Phân rã cảm xúc</div>
                    <div style='width: 100%; margin-top: 4px; margin-bottom: 6px;'>
                        <div style='display: flex; height: 10px; width: 100%; border-radius: 5px; overflow: hidden;'>
                            <div style='width: {pos_pct}%; background-color: #5CE488;' title='Tích cực {pos_pct:.0f}%'></div>
                            <div style='width: {neu_pct}%; background-color: #FBBF24;' title='Trung lập {neu_pct:.0f}%'></div>
                            <div style='width: {neg_pct}%; background-color: #FF6C6C;' title='Tiêu cực {neg_pct:.0f}%'></div>
                        </div>
                    </div>
                    <div style='display: flex; justify-content: space-between; font-size: 11px; color: #A1A1AA;'>
                        <span>🟢 {pos_pct:.0f}%</span>
                        <span>🟡 {neu_pct:.0f}%</span>
                        <span>🔴 {neg_pct:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")  # Margin spacing
    
            # Competitor rankings benchmark (Only shown when "Tất cả" is selected)
            if selected_app_name == "Tất cả" and app_col and app_col in filtered_df.columns:
                st.markdown("### So sánh Chỉ số Hài lòng giữa các Ngân hàng")
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    with st.container(height=350, border=True):
                        st.markdown("#### Chỉ số cảm xúc ròng (NSS) theo Ngân hàng")
                        def calc_nss_group(group):
                            total = len(group)
                            if total == 0: return 0.0
                            pos = (group == 'Tích cực').sum()
                            neg = (group == 'Tiêu cực').sum()
                            return ((pos - neg) / total) * 100
                        
                        bank_nss = filtered_df.groupby(app_col)['Sentiment'].apply(calc_nss_group).reset_index(name='NSS')
                        bank_nss = bank_nss.sort_values('NSS', ascending=True)
                        bank_nss['Color'] = bank_nss['NSS'].apply(lambda x: '#5CE488' if x >= 0 else '#FF6C6C')
                        
                        fig_bank_nss = px.bar(bank_nss, y=app_col, x='NSS', orientation='h',
                                             template='plotly_dark')
                        fig_bank_nss.update_traces(marker_color=bank_nss['Color'])
                        fig_bank_nss.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title="Chỉ số NSS (%)", gridcolor='#2C2C30'),
                            yaxis=dict(title=None),
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=240
                        )
                        st.plotly_chart(fig_bank_nss, width="stretch")
                with comp_col2:
                    with st.container(height=350, border=True):
                        st.markdown("#### Điểm hài lòng CSAT trung bình theo Ngân hàng")
                        bank_csat = filtered_df.groupby(app_col)['CSAT_Score'].mean().reset_index(name='CSAT')
                        bank_csat = bank_csat.sort_values('CSAT', ascending=True)
                        
                        fig_bank_csat = px.bar(bank_csat, y=app_col, x='CSAT', orientation='h',
                                              template='plotly_dark')
                        fig_bank_csat.update_traces(marker_color='#6366F1')
                        fig_bank_csat.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title="Điểm CSAT (1-5)", gridcolor='#2C2C30'),
                            yaxis=dict(title=None),
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=240
                        )
                        st.plotly_chart(fig_bank_csat, width="stretch")
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Row 2: Trend & distribution charts dynamically updated by OLAP controls
            r2_col1, r2_col2 = st.columns([1.5, 1])
            with r2_col1:
                with st.container(height=480, border=True):
                    st.markdown(f"#### Xu hướng {analysis_attrib.lower().replace('nss', 'NSS').replace('csat', 'CSAT')} theo {breakdown_dim_label.lower()}")
                    if date_col and date_col in filtered_df.columns:
                        df_time = filtered_df.copy()
                        
                        if selected_dim_col == "Aspects":
                            df_time = df_time.explode("Aspects")
                            
                        date_range_days = (df_time[date_col].max() - df_time[date_col].min()).days if not df_time.empty else 0
                        if date_range_days <= 14:
                            df_time['TimeGroup'] = df_time[date_col].dt.strftime('%d/%m')
                        else:
                            df_time['TimeGroup'] = df_time[date_col].dt.strftime('T%m').apply(lambda x: str(x).replace('T0', 'T'))
                        
                        if not df_time.empty:
                            def calc_nss(group):
                                total = len(group)
                                if total == 0:
                                    return 0.0
                                pos = (group == 'Tích cực').sum()
                                neg = (group == 'Tiêu cực').sum()
                                return ((pos - neg) / total) * 100
    
                            if analysis_attrib == "Số lượng đánh giá":
                                trend_data = df_time.groupby(['TimeGroup', selected_dim_col]).size().reset_index(name='Số lượng')
                                fig_trend = px.bar(trend_data, x='TimeGroup', y='Số lượng', color=selected_dim_col,
                                                   barmode='stack',
                                                   template='plotly_dark',
                                                   **color_args)
                                fig_trend.update_layout(yaxis=dict(title="Số lượng reviews"))
                            elif analysis_attrib == "Điểm CSAT trung bình":
                                trend_data = df_time.groupby(['TimeGroup', selected_dim_col])['CSAT_Score'].mean().reset_index(name='CSAT trung bình')
                                fig_trend = px.line(trend_data, x='TimeGroup', y='CSAT trung bình', color=selected_dim_col,
                                                    template='plotly_dark',
                                                    **color_args)
                                fig_trend.update_layout(yaxis=dict(title="CSAT trung bình"))
                            else: # Chỉ số NSS
                                trend_data = df_time.groupby(['TimeGroup', selected_dim_col])['Sentiment'].apply(calc_nss).reset_index(name='Chỉ số NSS')
                                fig_trend = px.line(trend_data, x='TimeGroup', y='Chỉ số NSS', color=selected_dim_col,
                                                    template='plotly_dark',
                                                    **color_args)
                                fig_trend.update_layout(yaxis=dict(title="Chỉ số NSS (%)"))
                                
                            fig_trend.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color="#A1A1AA", family="Plus Jakarta Sans"),
                                xaxis=dict(title=None, gridcolor='#2C2C30'),
                                yaxis=dict(gridcolor='#2C2C30'),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=None, entrywidth=0.45, entrywidthmode="fraction"),
                                margin=dict(t=40, b=40, l=10, r=10),
                                height=360
                            )
                            st.plotly_chart(fig_trend, width="stretch")
                        else:
                            st.info("Không có dữ liệu trong khoảng thời gian này.")
                    else:
                        st.info("Vui lòng tải dữ liệu chứa cột thời gian.")
            with r2_col2:
                with st.container(height=480, border=True):
                    st.markdown(f"#### Tỷ trọng đánh giá theo {breakdown_dim_label.lower()}")
                    
                    if selected_dim_col == "Aspects":
                        dist_df = filtered_df.explode('Aspects')
                    else:
                        dist_df = filtered_df.copy()
                    
                    if not dist_df.empty and selected_dim_col in dist_df.columns:
                        if analysis_attrib == "Số lượng đánh giá":
                            dist_data = dist_df[selected_dim_col].value_counts().reset_index()
                            dist_data.columns = [selected_dim_col, 'count']
                            
                            fig_pie = px.pie(dist_data, values='count', names=selected_dim_col,
                                             color=selected_dim_col,
                                             hole=0.6,
                                             template='plotly_dark',
                                             **color_args)
                            fig_pie.update_layout(
                                showlegend=True,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=40, b=10, l=10, r=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=None, entrywidth=0.45, entrywidthmode="fraction"),
                                height=360
                            )
                            total_val = dist_data['count'].sum()
                            fig_pie.add_annotation(
                                text=f"<b>{total_val:,}</b><br><span style='font-size:11px;color:#A1A1AA;'>tổng cộng</span>",
                                showarrow=False,
                                font=dict(size=16, color="#FFFFFF", family="Plus Jakarta Sans")
                            )
                            st.plotly_chart(fig_pie, width="stretch")
                        else:
                            def calc_nss(group):
                                total = len(group)
                                if total == 0:
                                    return 0.0
                                pos = (group == 'Tích cực').sum()
                                neg = (group == 'Tiêu cực').sum()
                                return ((pos - neg) / total) * 100
    
                            if analysis_attrib == "Điểm CSAT trung bình":
                                val_col = 'CSAT_Score'
                                agg_func = 'mean'
                                y_title = "CSAT trung bình"
                                fmt = ".2f"
                            else:
                                val_col = 'Sentiment'
                                agg_func = calc_nss
                                y_title = "Chỉ số NSS (%)"
                                fmt = ".1f"
                                
                            agg_series = dist_df.groupby(selected_dim_col)[val_col].agg(agg_func)
                            agg_series.name = 'value'
                            dist_data = agg_series.reset_index()
                            dist_data = dist_data.sort_values('value', ascending=False)
                            
                            fig_bar = px.bar(dist_data, x=selected_dim_col, y='value',
                                             color=selected_dim_col,
                                             template='plotly_dark',
                                             **color_args)
                            fig_bar.update_layout(
                                showlegend=False,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(title=None, gridcolor='#2C2C30'),
                                yaxis=dict(title=y_title, gridcolor='#2C2C30'),
                                margin=dict(t=10, b=80, l=10, r=10),
                                height=360
                            )
                            st.plotly_chart(fig_bar, width="stretch")
                    else:
                        st.info("Không có dữ liệu thỏa mãn bộ lọc.")
                        
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Row 3: Stacked Progress Bar comparing categories of the selected dimension
            if selected_dim_col == "Aspects":
                comp_df = filtered_df.explode('Aspects')
            else:
                comp_df = filtered_df.copy()
                
            if selected_dim_col and selected_dim_col in comp_df.columns:
                if selected_dim_col == 'Sentiment':
                    counts = comp_df['Sentiment'].value_counts()
                    pivot_comp = pd.DataFrame(0, index=['Tích cực', 'Trung lập', 'Tiêu cực'], columns=['Tích cực', 'Trung lập', 'Tiêu cực'])
                    pivot_comp.index.name = selected_dim_col
                    for col in pivot_comp.index:
                        if col in counts.index:
                            pivot_comp.loc[col, col] = counts[col]
                else:
                    group_sentiment = comp_df.groupby([selected_dim_col, 'Sentiment']).size().reset_index(name='count')
                    pivot_comp = group_sentiment.pivot(index=selected_dim_col, columns='Sentiment', values='count').fillna(0)
                
                for col in ['Tích cực', 'Trung lập', 'Tiêu cực']:
                    if col not in pivot_comp.columns:
                        pivot_comp[col] = 0
                        
                pivot_comp['Total'] = pivot_comp['Tích cực'] + pivot_comp['Trung lập'] + pivot_comp['Tiêu cực']
                pivot_comp = pivot_comp[pivot_comp['Total'] > 0]
                
                # Create a horizontal stacked bar chart
                pivot_pct = pivot_comp[['Tích cực', 'Trung lập', 'Tiêu cực']].copy()
                row_sums = pivot_pct.sum(axis=1)
                for col in pivot_pct.columns:
                    pivot_pct[col] = (pivot_pct[col] / row_sums * 100).round(1)
                
                pivot_pct = pivot_pct.reset_index()
                melted_comp = pivot_pct.melt(id_vars=selected_dim_col, value_vars=['Tích cực', 'Trung lập', 'Tiêu cực'], var_name='Cảm xúc', value_name='Phần trăm (%)')
                
                # Sort categories by Positive percentage descending so when Plotly draws from top to bottom, the highest is at the top
                sorted_categories = pivot_pct.sort_values(by='Tích cực', ascending=False)[selected_dim_col].tolist()
                
                fig_comp = px.bar(melted_comp, y=selected_dim_col, x='Phần trăm (%)', color='Cảm xúc',
                                  orientation='h',
                                  barmode='stack',
                                  color_discrete_map=LABEL_COLORS,
                                  template='plotly_dark',
                                  category_orders={selected_dim_col: sorted_categories})
                                  
                fig_comp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(title=None),
                    xaxis=dict(title="Tỷ lệ cảm xúc (%)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=None, entrywidth=120, entrywidthmode="pixels"),
                    margin=dict(t=40, b=10, l=10, r=10),
                    height=380
                )
                # Bottom Row: Aspect comparison and Top Keywords side-by-side
                comp_r_col1, comp_r_col2 = st.columns([1.5, 1])
                with comp_r_col1:
                    with st.container(height=480, border=True):
                        st.markdown(f"#### So sánh tỷ lệ cảm xúc theo {breakdown_dim_label.lower()}")
                        st.plotly_chart(fig_comp, width="stretch")
                with comp_r_col2:
                    with st.container(height=480, border=True):
                        st.markdown("#### Từ khóa thảo luận nổi bật")
                        
                        pos_kws = extract_top_keywords(filtered_df, "Tích cực")
                        neg_kws = extract_top_keywords(filtered_df, "Tiêu cực")
                        
                        st.markdown("<p style='font-size: 13px; font-weight: 700; color: #5CE488; margin-top: 10px;'>🟢 ĐÁNH GIÁ TÍCH CỰC</p>", unsafe_allow_html=True)
                        if pos_kws:
                            pos_badges = "".join(f"<span style='background-color: #1a3926; color: #5CE488; border: 1px solid rgba(92, 228, 136, 0.3); padding: 4px 10px; margin: 4px; border-radius: 12px; display: inline-block; font-size: 12px; font-weight: 600;'>#{kw}</span>" for kw in pos_kws)
                            st.markdown(f"<div style='margin-bottom: 15px;'>{pos_badges}</div>", unsafe_allow_html=True)
                        else:
                            st.caption("Chưa đủ dữ liệu từ khóa.")
                            
                        st.markdown("<p style='font-size: 13px; font-weight: 700; color: #FF6C6C; margin-top: 10px;'>🔴 ĐÁNH GIÁ TIÊU CỰC</p>", unsafe_allow_html=True)
                        if neg_kws:
                            neg_badges = "".join(f"<span style='background-color: #412426; color: #FF6C6C; border: 1px solid rgba(255, 108, 108, 0.3); padding: 4px 10px; margin: 4px; border-radius: 12px; display: inline-block; font-size: 12px; font-weight: 600;'>#{kw}</span>" for kw in neg_kws)
                            st.markdown(f"<div>{neg_badges}</div>", unsafe_allow_html=True)
                        else:
                            st.caption("Chưa đủ dữ liệu từ khóa.")
            else:
                st.info("Không tìm thấy dữ liệu chiều phân tích.")

# ----------------- TAB 2: PHÂN TÍCH SÂU -----------------
with tab2:
    if c_df is None:
        st.markdown("""
        <div style='background-color: #1E1E22; border: 1px solid #2C2C30; border-radius: 12px; padding: 40px; text-align: center; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); margin-top: 20px;'>
            <h3 style='color: #FFFFFF; font-weight: 700; margin-bottom: 15px; font-family: "Plus Jakarta Sans";'>Chưa có dữ liệu phân tích</h3>
            <p style='color: #A1A1AA; font-size: 15px; line-height: 1.6; max-width: 600px; margin: 0 auto 25px auto; font-family: "Plus Jakarta Sans";'>
                Hệ thống phân tích cảm xúc khách hàng đối với các ứng dụng ngân hàng số sử dụng mô hình học sâu học máy tiếng Việt <b>ViSoBERT</b>.
            </p>
            <div style='display: inline-block; background-color: rgba(79, 70, 229, 0.05); border: 1px dashed #3F3F46; border-radius: 8px; padding: 15px 25px; color: #A1A1AA; font-weight: 600; font-size: 14px; font-family: "Plus Jakarta Sans";'>
                📂 Vui lòng tải lên file dữ liệu (.csv, .xlsx) ở thanh bên trái hoặc chọn "Sử dụng dữ liệu giả lập (Demo)" để bắt đầu!
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif filtered_df.empty:
        st.warning("Không tìm thấy kết quả thỏa mãn bộ lọc hiện tại.")
    else:
        has_labels = False
        if label_col and label_col in filtered_df.columns:
            eval_df = filtered_df.dropna(subset=[label_col])
            if not eval_df.empty:
                has_labels = True
            
        classes = ["Tiêu cực", "Trung lập", "Tích cực"]
        
        # Confusion matrix data computation
        if has_labels:
            eval_df = filtered_df.dropna(subset=[label_col])
            y_true_raw = eval_df[label_col]
            y_true = []
            for item in y_true_raw:
                if str(item).strip() in ['0', '1', '2']:
                    y_true.append(LABEL_MAPPING[int(item)])
                elif str(item).strip().lower() in ['tiêu cực', 'tieu cuc', 'negative', 'label_0']:
                    y_true.append("Tiêu cực")
                elif str(item).strip().lower() in ['trung lập', 'trung lap', 'neutral', 'label_1']:
                    y_true.append("Trung lập")
                else:
                    y_true.append("Tích cực")
                    
            y_pred = eval_df['Sentiment'].tolist()
            
            if len(y_true) > 0 and len(y_pred) > 0:
                try:
                    cm = confusion_matrix(y_true, y_pred, labels=["Tiêu cực", "Trung lập", "Tích cực"])
                    acc = accuracy_score(y_true, y_pred)
                    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    
                    f1s = f1_score(y_true, y_pred, average=None, labels=["Tiêu cực", "Trung lập", "Tích cực"], zero_division=0)
                    f1_neg, f1_neu, f1_pos = f1s[0], f1s[1], f1s[2]
                except Exception as e:
                    has_labels = False
            else:
                has_labels = False
                
        # HIỂN THỊ ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH (Confusion Matrix và KPI)
        # Chỉ tính toán và vẽ biểu đồ nếu tệp dữ liệu tải lên thực sự có sẵn cột Nhãn cảm xúc gốc
        if has_labels:
            col_eval1, col_eval2 = st.columns([1.3, 1])
            with col_eval1:
                with st.container(height=450, border=True):
                    st.markdown("#### Confusion Matrix")
                    fig_cm = px.imshow(cm,
                                       labels=dict(x="Nhãn dự đoán", y="Nhãn thực tế"),
                                       x=["Tiêu cực", "Trung lập", "Tích cực"],
                                       y=["Tiêu cực", "Trung lập", "Tích cực"],
                                       text_auto=True,
                                       color_continuous_scale="Blues",
                                       template="plotly_dark")
                    fig_cm.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        coloraxis_showscale=False,
                        margin=dict(t=20, b=20, l=20, r=20),
                        height=350
                    )
                    st.plotly_chart(fig_cm, width="stretch")
            with col_eval2:
                with st.container(height=450, border=True):
                    st.markdown("#### Chỉ số đánh giá")
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric(label="Macro F1-Score", value=f"{f1_macro:.3f}")
                    m_col2.metric(label="Accuracy", value=f"{acc * 100:.1f}%")
                    
                    m_col3, m_col4 = st.columns(2)
                    m_col3.metric(label="Precision", value=f"{prec_macro:.3f}")
                    m_col4.metric(label="Recall", value=f"{rec_macro:.3f}")
                    
                    st.markdown("**F1 theo từng lớp**")
                    st.progress(min(1.0, max(0.0, float(f1_pos))), text=f"Tích cực: {f1_pos:.2f}")
                    st.progress(min(1.0, max(0.0, float(f1_neg))), text=f"Tiêu cực: {f1_neg:.2f}")
                    st.progress(min(1.0, max(0.0, float(f1_neu))), text=f"Trung tính: {f1_neu:.2f}")
        else:
            # Ẩn Confusion Matrix và hiển thị thông báo hướng dẫn để giao diện minh bạch 100% khi không có nhãn gốc
            st.info("💡 Các chỉ số đánh giá mô hình (Confusion Matrix, Accuracy, F1-Score) chỉ hiển thị khi tệp dữ liệu tải lên có sẵn cột Nhãn cảm xúc gốc.")
            st.markdown("<br>", unsafe_allow_html=True)
                
        # Model comparisons
        model_comp_df = pd.DataFrame({
            "MÔ HÌNH": ["Naive Bayes", "SVM (Linear)", "Logistic Reg.", "PhoBERT (tốt nhất)"],
            "ACCURACY": ["71.4%", "76.8%", "74.2%", "88.2%"],
            "MACRO F1": ["0.643", "0.714", "0.691", "0.834"],
            "PRECISION": ["0.661", "0.728", "0.703", "0.829"],
            "RECALL": ["0.628", "0.701", "0.680", "0.841"],
            "THỜI GIAN": ["< 1ms", "< 1ms", "< 1ms", "~0.5s"]
        })
        with st.container(border=True):
            st.markdown("#### So sánh mô hình")
            st.dataframe(model_comp_df, hide_index=True, width="stretch")
        
        # Raw search & tables drill down
        with st.container(border=True):
            st.markdown("#### Reviews đã phân tích")
            
            # Filters row
            col_tbl_f1, col_tbl_f2, col_tbl_f3 = st.columns([2, 1, 1])
            with col_tbl_f1:
                search_query = st.text_input("Tìm kiếm review:", placeholder="Nhập từ khóa tìm kiếm...", key="raw_tbl_search")
            with col_tbl_f2:
                lbl_filter = st.selectbox("Lọc theo nhãn:", ["Tất cả nhãn", "Tích cực", "Trung lập", "Tiêu cực"], key="raw_tbl_lbl")
            with col_tbl_f3:
                if app_col and app_col in filtered_df.columns:
                    bank_options = ["Tất cả app"] + sorted(filtered_df[app_col].dropna().unique().tolist())
                else:
                    bank_options = ["Tất cả app"]
                bank_filter = st.selectbox("Lọc theo ngân hàng:", bank_options, key="raw_tbl_bank")
                
            tbl_df = filtered_df.copy()
            if search_query:
                tbl_df = tbl_df[tbl_df[txt_col].str.contains(search_query, case=False, na=False)]
            if lbl_filter != "Tất cả nhãn":
                tbl_df = tbl_df[tbl_df['Sentiment'] == lbl_filter]
            if bank_filter != "Tất cả app" and app_col and app_col in tbl_df.columns:
                tbl_df = tbl_df[tbl_df[app_col] == bank_filter]
                
            total_items = len(tbl_df)
            page_size = 5
            total_pages = max(1, int(np.ceil(total_items / page_size)))
            
            # Session state initialization for page number
            if 'raw_tbl_page' not in st.session_state:
                st.session_state['raw_tbl_page'] = 1
                
            # Boundary check if filters change
            if st.session_state['raw_tbl_page'] > total_pages:
                st.session_state['raw_tbl_page'] = total_pages
            if st.session_state['raw_tbl_page'] < 1:
                st.session_state['raw_tbl_page'] = 1
                
            page_num = st.session_state['raw_tbl_page']
            
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, total_items)
            page_items = tbl_df.iloc[start_idx:end_idx]
            
            table_data = []
            for idx, row in page_items.iterrows():
                text_val = str(row[txt_col])
                if len(text_val) > 85:
                    text_val = text_val[:85] + "..."
                bank_val = row.get(app_col, "N/A")
                sent_val = row['Sentiment']
                
                prob = row.get('Probabilities', [0, 0, 0])
                if isinstance(prob, list):
                    if sent_val == "Tiêu cực": conf = prob[0]
                    elif sent_val == "Trung lập": conf = prob[1]
                    else: conf = prob[2]
                else:
                    conf = 0.85
                
                dt_val = row.get(date_col, datetime.datetime.now())
                if isinstance(dt_val, pd.Timestamp):
                    dt_str = dt_val.strftime("%d/%m/%Y %H:%M")
                else:
                    dt_str = str(dt_val)
                    
                table_data.append({
                    "Nội dung review": text_val,
                    "Ứng dụng": bank_val,
                    "Cảm xúc": sent_val,
                    "Độ tin cậy": float(conf) * 100,
                    "Thời gian": dt_str
                })
                
            display_df = pd.DataFrame(table_data)
            
            if not display_df.empty:
                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Cảm xúc": st.column_config.TextColumn(
                            "Cảm xúc",
                            help="Cảm xúc dự đoán bởi mô hình"
                        ),
                        "Độ tin cậy": st.column_config.ProgressColumn(
                            "Độ tin cậy",
                            help="Độ tin cậy của mô hình",
                            format="%.0f%%",
                            min_value=0.0,
                            max_value=100.0
                        )
                    }
                )
                
                # Bottom elements row: Display status on left, custom pagination on right
                col_bottom_left, col_bottom_right = st.columns([2.5, 1.5])
                with col_bottom_left:
                    st.markdown(f"<p style='color: #A1A1AA; font-size: 13px; margin-top: 8px;'>Hiển thị {start_idx+1}-{end_idx} trên tổng số {total_items} reviews</p>", unsafe_allow_html=True)
                with col_bottom_right:
                    # Pagination: (-) 1 (+) layout
                    col_prev, col_num, col_next = st.columns([1, 2, 1])
                    with col_prev:
                        if st.button("-", key="prev_page_btn", width="stretch", disabled=(page_num <= 1)):
                            st.session_state['raw_tbl_page'] = max(1, page_num - 1)
                            st.rerun()
                    with col_num:
                        st.markdown(f"<p style='text-align: center; font-size: 14px; font-weight: 700; margin-top: 6px; color: #FFFFFF;'>Trang {page_num} / {total_pages}</p>", unsafe_allow_html=True)
                    with col_next:
                        if st.button("+", key="next_page_btn", width="stretch", disabled=(page_num >= total_pages)):
                            st.session_state['raw_tbl_page'] = min(total_pages, page_num + 1)
                            st.rerun()
            else:
                st.info("Không có dữ liệu hiển thị.")

# ----------------- TAB 3: DEMO LIVE -----------------
with tab3:
    col_demo_left, col_demo_right = st.columns([1.2, 1])
    
    if 'demo_history' not in st.session_state:
        st.session_state['demo_history'] = [
            {"text": "Ứng dụng hay bị lag, đăng nhập mãi không được, nhiều lúc tự thoát ra ngoài rất bực mình. Giao diện thì khó dùng,", "sentiment": "Tiêu cực", "time": "vừa xong"},
            {"text": "App rất nhanh, chuyển tiền tiện lợi, giao diện đẹp và dễ dùng.", "sentiment": "Tích cực", "time": "2 phút"},
            {"text": "OTP không gửi được, chờ mãi không thấy, rất bực...", "sentiment": "Tiêu cực", "time": "5 phút"}
        ]
        
    with col_demo_left:
        with st.container(border=True):
            st.markdown("#### Nhập review")
            
            if 'demo_input' not in st.session_state:
                st.session_state['demo_input'] = ""
                
            sample_options = ["-- Nhập câu mới --"] + [item["text"] for item in st.session_state['demo_history']]
            selected_hist = st.selectbox("Chọn câu mẫu:", sample_options, key="demo_sample")
            
            if selected_hist != "-- Nhập câu mới --":
                st.session_state['demo_input'] = selected_hist
                
            user_input = st.text_area("Nội dung đánh giá:", 
                                      height=140,
                                      max_chars=512,
                                      placeholder="Gõ bình luận cần phân tích cảm xúc...",
                                      key="demo_input")
            
            st.caption(f"Độ dài: {len(user_input)} / 512 ký tự")
            
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                clear_btn = st.button("Xóa", key="demo_clear_btn")
            with col_btn2:
                analyze_btn = st.button("Phân tích", key="demo_run_btn")
                
            if clear_btn:
                st.session_state['demo_input'] = ""
                st.rerun()
                
            if analyze_btn and user_input.strip():
                # Add to history if unique
                if user_input.strip() not in [item["text"] for item in st.session_state['demo_history']]:
                    st.session_state['demo_history'].insert(0, {"text": user_input.strip(), "sentiment": "Đang tính...", "time": "vừa xong"})
                    
        # History list
        with st.container(border=True):
            st.markdown("#### Lịch sử phân tích")
            hist_data = []
            for item in st.session_state['demo_history'][:4]:
                t_text = item["text"]
                if len(t_text) > 60:
                    t_text = t_text[:60] + "..."
                hist_data.append({
                    "Nội dung": t_text,
                    "Cảm xúc": item["sentiment"],
                    "Thời gian": item["time"]
                })
            if hist_data:
                st.dataframe(pd.DataFrame(hist_data), width="stretch", hide_index=True)
            else:
                st.info("Chưa có lịch sử phân tích.")
        
    with col_demo_right:
        if analyze_btn and user_input.strip():
            if not model_loaded:
                st.error("Mô hình chưa được nạp.")
            else:
                words, importances, pred_class = explain_text_lime(user_input, tokenizer, model)
                pred_label = LABEL_MAPPING[pred_class]
                
                # Update history element
                for item in st.session_state['demo_history']:
                    if item["text"] == user_input.strip():
                        item["sentiment"] = pred_label
                
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                
                p_neg, p_neu, p_pos = probs[0], probs[1], probs[2]
                
                # Predict output KPI and bars using native components
                with st.container(border=True):
                    st.markdown("#### Kết quả dự đoán")
                    
                    if pred_label == "Tích cực":
                        st.success(f"**Kết quả: Tích cực**", icon="🟢")
                    elif pred_label == "Trung lập":
                        st.warning(f"**Kết quả: Trung lập**", icon="🟡")
                    else:
                        st.error(f"**Kết quả: Tiêu cực**", icon="🔴")
                        
                    c_neg, c_neu, c_pos = st.columns(3)
                    c_neg.metric(label="Tiêu cực", value=f"{p_neg*100:.0f}%")
                    c_neu.metric(label="Trung tính", value=f"{p_neu*100:.0f}%")
                    c_pos.metric(label="Tích cực", value=f"{p_pos*100:.0f}%")
                    
                    st.markdown("**Độ tin cậy cảm xúc:**")
                    st.progress(float(p_neg), text=f"Tiêu cực: {p_neg*100:.0f}%")
                    st.progress(float(p_neu), text=f"Trung tính: {p_neu*100:.0f}%")
                    st.progress(float(p_pos), text=f"Tích cực: {p_pos*100:.0f}%")
                
                # LIME explanation card
                with st.container(border=True):
                    st.markdown("#### Giải thích mô hình (XAI - LIME)")
                    st.caption("Các từ được tô màu Xanh lá (🟢) thuộc nhãn Tích cực, màu Vàng (🟡) thuộc nhãn Trung lập, màu Đỏ (🔴) thuộc nhãn Tiêu cực (chỉ số mũ thể hiện trọng số ảnh hưởng)")
                    
                    # Highlighted words (clean single line formatting without indentation)
                    html_elements = []
                    abs_importances = [abs(imp) for imp in importances]
                    max_imp = max(abs_importances) if abs_importances and max(abs_importances) > 0 else 1.0
                    
                    top_features = []
                    for w, imp in zip(words, importances):
                        top_features.append((w, imp))
                    top_features = sorted(top_features, key=lambda x: abs(x[1]), reverse=True)[:5]
                    
                    for w, imp in zip(words, importances):
                        alpha = min(0.7, (abs(imp) / max_imp) * 0.6) if max_imp > 0 else 0.0
                        
                        if abs(imp) <= 0.001:
                            bg_color = "transparent"
                            text_color = "#FFFFFF"
                            w_display = w
                        else:
                            # Determine semantic influence based on predicted class C:
                            # If pred_class is Tích cực (2): imp > 0 means Tích cực (Green), imp < 0 means Tiêu cực (Red)
                            # If pred_class is Tiêu cực (0): imp > 0 means Tiêu cực (Red), imp < 0 means Tích cực (Green)
                            # If pred_class is Trung lập (1): imp > 0 means Trung lập (Amber), imp < 0 is other/neutral
                            if pred_class == 2:
                                is_pos = (imp > 0)
                            elif pred_class == 0:
                                is_pos = (imp < 0)
                            else:
                                is_pos = None
                            
                            if is_pos is True:
                                bg_color = f"rgba(92, 228, 136, {alpha + 0.25})"
                                text_color = "#5CE488"
                                sign = "+" if imp > 0 else ""
                                w_display = f"{w}<sup style='font-size: 8px; color: #5CE488;'>{sign}{imp:.2f}</sup>"
                            elif is_pos is False:
                                bg_color = f"rgba(255, 108, 108, {alpha + 0.25})"
                                text_color = "#FF6C6C"
                                sign = "+" if imp > 0 else ""
                                w_display = f"{w}<sup style='font-size: 8px; color: #FF6C6C;'>{sign}{imp:.2f}</sup>"
                            elif pred_class == 1 and imp > 0:
                                bg_color = f"rgba(251, 191, 36, {alpha + 0.25})"
                                text_color = "#FBBF24"
                                w_display = f"{w}<sup style='font-size: 8px; color: #FBBF24;'>+{imp:.2f}</sup>"
                            else:
                                bg_color = f"rgba(100, 116, 139, {alpha + 0.25})"
                                text_color = "#94A3B8"
                                w_display = f"{w}<sup style='font-size: 8px; color: #94A3B8;'>{imp:.2f}</sup>"
                            
                        html_elements.append(f'<span style="background-color: {bg_color}; color: {text_color}; padding: 4px 6px; margin: 2px; border-radius: 4px; display: inline-block; font-weight: 600; font-size: 13.5px;">{w_display}</span>')
                    
                    html_text = "".join(html_elements).strip().replace("\n", "")
                    
                    # Render the highlighted text without spaces to avoid markdown code-block bug
                    st.markdown(f'<div style="background-color: #151518; border: 1px solid #2C2C30; border-radius: 8px; padding: 16px; line-height: 2.2; margin-bottom: 20px;">{html_text}</div>', unsafe_allow_html=True)
                    
                    # Plot LIME feature weights
                    if top_features:
                        top_df = pd.DataFrame(top_features, columns=["Từ khóa", "Trọng số"])
                        top_df["AbsTrọng số"] = top_df["Trọng số"].abs()
                        top_df = top_df.sort_values(by="AbsTrọng số", ascending=True)
                        
                        def get_influence_label(weight, p_class):
                            if p_class == 2:
                                return "Đẩy sang Tích cực" if weight > 0 else "Đẩy sang Tiêu cực"
                            elif p_class == 0:
                                return "Đẩy sang Tiêu cực" if weight > 0 else "Đẩy sang Tích cực"
                            elif p_class == 1:
                                return "Đẩy sang Trung lập" if weight > 0 else "Đẩy sang cảm xúc khác"
                            return "Khác"
                            
                        top_df["Ảnh hưởng"] = top_df["Trọng số"].apply(lambda x: get_influence_label(x, pred_class))
                        
                        fig_lime = px.bar(
                            top_df,
                            x="Trọng số",
                            y="Từ khóa",
                            orientation="h",
                            color="Ảnh hưởng",
                            color_discrete_map={
                                "Đẩy sang Tích cực": "#5CE488",
                                "Đẩy sang Tiêu cực": "#FF6C6C",
                                "Đẩy sang Trung lập": "#FBBF24",
                                "Đẩy sang cảm xúc khác": "#64748B",
                                "Khác": "#64748B"
                            },
                            template="plotly_dark"
                        )
                        fig_lime.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(title="Trọng số ảnh hưởng", gridcolor='#2C2C30'),
                            yaxis=dict(title=None),
                            showlegend=False,
                            margin=dict(t=10, b=10, l=10, r=10),
                            height=220
                        )
                        
                        st.markdown("**Top từ khóa tác động:**")
                        st.plotly_chart(fig_lime, width="stretch")
        else:
            st.info("Nhập văn bản bên trái rồi nhấn Phân tích.")

# ----------------- TAB 4: GIỚI THIỆU -----------------
with tab4:
    with st.container(border=True):
        st.markdown("### Hướng dẫn sử dụng Dashboard")
        st.markdown(
            "1. **Tải dữ liệu**: Tải tệp đánh giá (.csv, .xlsx) ở thanh bên trái hoặc chọn 'Sử dụng dữ liệu giả lập (Demo)'.\n"
            "2. **Khai báo cột**: Mở mục 'Chọn cột phân tích' ở đầu trang để kiểm tra và liên kết các cột dữ liệu tương ứng trong file của bạn (Nội dung đánh giá, Tên ngân hàng, Ngày đánh giá, Nhãn cảm xúc gốc).\n"
            "3. **Phân tích**: Sử dụng các Tab 'Tổng quan', 'Phân tích sâu' để giám sát chỉ số và Tab 'Demo Live' để thử nghiệm suy luận thời gian thực."
        )
        
        st.markdown("<hr style='margin: 18px 0; border-color: #2C2C30;'>", unsafe_allow_html=True)
        
        st.markdown("### Thông tin mô hình ViSoBERT")
        st.markdown(
            "Mô hình học sâu được tinh chỉnh (fine-tune) đặc thù cho bài toán phân tích cảm xúc khách hàng trong lĩnh vực ngân hàng số tại Việt Nam.\n\n"
            "- **Kiến trúc nền tảng**: ViSoBERT (Vietnamese Social Media BERT).\n"
            "- **Dữ liệu huấn luyện**: 17.967 đánh giá của người dùng thực tế trên Google Play Store (đã được làm sạch, chuẩn hóa NLP tiếng Việt và gán nhãn cảm xúc).\n"
            "- **Phân tích khía cạnh (Aspects)**: Trích xuất các khía cạnh trải nghiệm (Bảo mật, Giao dịch, Phí dịch vụ, Chăm sóc khách hàng...) dựa trên bộ luật từ khóa đặc trưng.\n"
            "- **Trí tuệ nhân tạo giải thích được (XAI)**: Tích hợp thư viện LIME để trực quan hóa trọng số đóng góp của các từ khóa vào quyết định phân loại của mô hình."
        )
