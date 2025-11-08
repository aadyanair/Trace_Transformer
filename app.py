# ============================================================
# 🧠 FTTransformer Dashboard App (Streamlit)
# ============================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import streamlit as st

# ============================================================
# 📊 MODEL ARCHITECTURE DEFINITION
# ============================================================

class FTTransformer(nn.Module):
    def __init__(self, num_features=107, num_classes=7, dim=256, depth=6, heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Feature embedding
        self.feature_embedding = nn.Linear(num_features, dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        x = self.feature_embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.classifier(x)
        return x


# ============================================================
# ⚙️ MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(weights_path="best_model.pth"):
    model = FTTransformer(num_features=107, num_classes=7, dim=256)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print("✅ Model weights loaded successfully!")
    else:
        print("Using initialized model.")
    model.eval()
    return model


# ============================================================
# 📂 CSV FILE HANDLING
# ============================================================

def load_input_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    if df.shape[1] != 107:
        st.error(f"❌ Expected 107 features, but got {df.shape[1]} columns in CSV.")
        return None
    return torch.FloatTensor(df.values), df


# ============================================================
# 🔮 PREDICTION FUNCTION
# ============================================================

def predict_from_csv(model, csv_file, class_labels):
    inputs, raw_df = load_input_csv(csv_file)
    if inputs is None:
        return None

    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    result_df = pd.DataFrame(probs, columns=class_labels)
    result_df["Predicted Class"] = [class_labels[i] for i in preds]
    return result_df


# ============================================================
# 🌐 STREAMLIT APP UI
# ============================================================

st.set_page_config(page_title="FTTransformer Dashboard", layout="wide")
st.title("🧠 Network Forensics Attack Classification Dashboard")
st.markdown("Upload a CSV file containing **107 network traffic features** to classify each record into one of 7 attack categories.")

# Load model
model = load_model("best_model.pth")

# Define real class labels
class_labels = [
    "Benign",
    "Bot",
    "DDoS-Attack-HOIC",
    "DDoS-Attack-LOIC-UDP",
    "Dos Attack-Hulk",
    "Dos Attack-SlowHTTPTest",
    "Infiltration"    
]

# File uploader
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    result_df = predict_from_csv(model, uploaded_file, class_labels)
    
    if result_df is not None:
        st.subheader("🔍 Prediction Results")
        st.dataframe(
            result_df.style.highlight_max(
                subset=class_labels, axis=1, color="#A8DADC"
            )
        )
        
        # Show class distribution chart
        st.subheader("📈 Predicted Class Distribution")
        chart_data = result_df["Predicted Class"].value_counts().reset_index()
        chart_data.columns = ["Class", "Count"]
        st.bar_chart(chart_data.set_index("Class"))
else:
    st.info("📎 Please upload a CSV file to start predictions.")
