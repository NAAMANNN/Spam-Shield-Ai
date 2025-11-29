import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Spam Shield AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f5f7f9;
    }
    
    /* Custom Title */
    .title-text {
        font-size: 50px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Card-like containers */
    .stTextArea > div {
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Result Boxes */
    .spam-box {
        padding: 20px;
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        color: #991b1b;
        border-radius: 5px;
        font-weight: bold;
        margin-top: 20px;
    }
    
    .safe-box {
        padding: 20px;
        background-color: #dcfce7;
        border-left: 5px solid #22c55e;
        color: #166534;
        border-radius: 5px;
        font-weight: bold;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Loads data from CSV or creates dummy data for simulation.
    """
    status = "⚠️ Simulation (Dummy Data)"
    try:
        df = pd.read_csv('emails.csv')
        
        if df.shape[1] < 10:
             raise ValueError("CSV has too few columns")
             
        status = "✅ Real Dataset (emails.csv)"
        return df, status
        
    except (FileNotFoundError, ValueError, Exception):
        
        words = ['hello', 'win', 'prize', 'urgent', 'offer', 'money', 'free', 'click', 'meeting', 'project']
        spam_words = ['win', 'prize', 'urgent', 'offer', 'money', 'free', 'click']
        safe_words = ['hello', 'meeting', 'project']
        
        data = []
        for _ in range(200):
            is_spam = np.random.randint(0, 2)
            row = {}
            if is_spam == 1:
                for word in words:
                    if word in spam_words:
                        row[word] = 1 if np.random.random() > 0.1 else 0
                    else:
                        row[word] = 0
            else:
                for word in words:
                    if word in safe_words:
                        row[word] = 1 if np.random.random() > 0.1 else 0
                    else:
                        row[word] = 0
            row['Prediction'] = is_spam
            data.append(row)
            
        return pd.DataFrame(data), status


@st.cache_resource
def train_model(df, split_size):
   
    if 'Email No.' in df.columns:
        df = df.drop('Email No.', axis=1)
    elif 'Email No' in df.columns: 
        df = df.drop('Email No', axis=1)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-split_size), random_state=42)
    
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X, y, len(X_train), len(X_test)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621183.png", width=100)
    st.title("⚙️ Control Panel")
    
    df, data_source = load_data()
    
    if "Real" in data_source:
        st.success(f"Status: {data_source}")
    else:
        st.warning(f"Status: {data_source}")
        st.caption("Upload 'emails.csv' to improve accuracy.")

    st.markdown("### 🧠 Model Tuning")
    split_slider = st.slider("Training Data Split (%)", 10, 90, 80, 5)
    split_size = split_slider / 100.0

    model, accuracy, X, y, train_len, test_len = train_model(df, split_size)

    st.markdown("### 📊 Dataset Info")
    st.metric(label="Model Accuracy", value=f"{accuracy*100:.1f}%", delta="Live Update")
    st.info(f"Training on {train_len} samples")


st.markdown('<div class="title-text">🛡️ Spam Shield AI</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Advanced Machine Learning for Message Security</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🕵️‍♂️ Spam Detector", "📈 Data Analytics"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Message Analyzer")
        
        st.write("Try an example:")
        example_cols = st.columns(3)
        if example_cols[0].button("📝 Load Safe Email"):
            st.session_state['text_input'] = "Please review the attached report regarding the monthly contract discussion. Let me know if you have any questions."
        if example_cols[1].button("🚨 Load Spam Email"):
            st.session_state['text_input'] = "URGENT! You have won a FREE cash prize. Click here to claim your money now!"
        if example_cols[2].button("🗑️ Clear"):
            st.session_state['text_input'] = ""
            
        user_input = st.text_area("Paste your message here:", 
                                  value=st.session_state.get('text_input', ''),
                                  height=200,
                                  placeholder="Type something...")
        
        if st.button("Analyze Content", type="primary", use_container_width=True):
            if user_input:
                with st.spinner("Scanning for keywords..."):
                    time.sleep(0.5) 
                    
            
                    input_data = {word: 0 for word in X.columns}
                    
                    clean_text = re.sub(r'[^\w\s]', ' ', user_input) 
                    words = clean_text.split()
                    
                   
                    matched_count = 0
                    for word in words:
                        word = word.lower() 
                        if word in input_data:
                            input_data[word] += 1
                            matched_count += 1
                    
            
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)
                    
                    st.write("---")
                    
                    if matched_count == 0:
                        st.warning("⚠️ Unknown Vocabulary: The model didn't recognize any words in your message.")
                    
                    if prediction[0] == 1:
                        st.markdown(f"""
                        <div class="spam-box">
                            <h2>🚫 SPAM DETECTED</h2>
                            <p>This message contains patterns commonly associated with junk mail or scams.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-box">
                            <h2>✅ MESSAGE SAFE</h2>
                            <p>This message appears to be legitimate.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
            else:
                st.warning("Please enter text first.")

    with col2:
        st.markdown("### How it works")
        st.info("""
        **1. Tokenization:** Splits your sentence into individual words.
        
        **2. Filtering:** Removes punctuation and symbols.
        
        **3. Vectorization:** Maps your words against known dictionary words.
        
        **4. SVM Classification:** Uses math to draw a line between 'Safe' and 'Spam'.
        """)

with tab2:
    st.header("📊 Behind the Scenes")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Dataset Balance")
        chart_data = y.value_counts().rename(index={0: 'Safe', 1: 'Spam'})
        st.bar_chart(chart_data, color=["#22c55e"])
        st.caption("Distribution of Safe vs Spam emails in training data")
        
    with col_b:
        st.subheader("Model Configuration")
        st.write(model)
        st.markdown(f"**Kernel Type:** {model.kernel}")
        st.markdown(f"**Training Samples:** {train_len}")
        st.markdown(f"**Testing Samples:** {test_len}")
