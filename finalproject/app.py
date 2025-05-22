import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import google.generativeai as genai
import os

# 設定 Gemini API 金鑰
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# 初始化 Gemini 模型
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# 表情對應提示語
emotion_prompts = {
    "happy": "顧客感到開心，請給出 3 句語氣溫和且富有人情味的客服回應語句。",
    "sad": "顧客感到難過，請給出 3 句溫暖且具有安撫意味的客服回應語句。",
    "angry": "顧客感到生氣，請給出 3 句有同理心且願意協助的客服回應語句。",
    "fear": "顧客感到害怕，請給出 3 句讓人安心與具備協助態度的客服回應語句。",
    "surprise": "顧客感到驚訝，請給出 3 句親切且不失專業的客服回應語句。",
    "disgust": "顧客感到反感，請給出 3 句展現理解並希望改進的客服回應語句。",
    "neutral": "顧客表情中性，請給出 3 句友善問候式的客服回應語句。"
}

# Streamlit 介面
st.title("📸 AI臉部情緒辨識即時應對系統")

# 根據部署環境使用不同方式取得影像
if os.environ.get("STREAMLIT_SERVER_HEADLESS") == "1":
    uploaded_file = st.file_uploader("請上傳一張臉部圖片（jpg/png）", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("請使用攝影機拍攝圖片")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    try:
        # 表情辨識
        analysis = DeepFace.analyze(img_array, actions=["emotion"], enforce_detection=False)
        dominant_emotion = analysis[0]["dominant_emotion"]
        st.markdown(f"🔍 **偵測到的情緒為：{dominant_emotion}**")

        # Gemini 語句產生
        if dominant_emotion in emotion_prompts:
            prompt = emotion_prompts[dominant_emotion]
            response = model.generate_content(prompt)
            if hasattr(response, "text"):
                suggestions = response.text.strip().split("\n")
                st.markdown("💬 **建議回應語句：**")
                for idx, s in enumerate(suggestions, 1):
                    if s.strip():
                        st.write(f"{idx}. {s.strip('- ').strip()}")
            else:
                st.error("⚠️ 無法取得 Gemini 回應內容。")
        else:
            st.warning("⚠️ 未定義的情緒類別。")
    except Exception as e:
        st.error(f"❌ 錯誤：{e}")
