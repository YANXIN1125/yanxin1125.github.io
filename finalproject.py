import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import google.generativeai as genai

# 設定 Gemini API 金鑰
genai.configure(api_key="AIzaSyAgYRoDVatXcHZejL7rfYmBa4yoKu613wk")

# 初始化模型
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# 表情對應敘述
emotion_prompts = {
    "happy": "顧客感到開心，請給出 3 句語氣溫和且富有人情味的客服回應語句。",
    "sad": "顧客感到難過，請給出 3 句溫暖且具有安撫意味的客服回應語句。",
    "angry": "顧客感到生氣，請給出 3 句有同理心且願意協助的客服回應語句。",
    "fear": "顧客感到害怕，請給出 3 句讓人安心與具備協助態度的客服回應語句。",
    "surprise": "顧客感到驚訝，請給出 3 句親切且不失專業的客服回應語句。",
    "disgust": "顧客感到反感，請給出 3 句展現理解並希望改進的客服回應語句。",
    "neutral": "顧客表情中性，請給出 3 句友善問候式的客服回應語句。"
}

# Streamlit UI
st.title("📸 AI臉部情緒辨識即時應對系統")
uploaded_file = st.camera_input("監視器")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    try:
        # 表情辨識
        analysis = DeepFace.analyze(img_array, actions=["emotion"], enforce_detection=False)
        dominant_emotion = analysis[0]["dominant_emotion"]
        st.markdown(f"🔍 **偵測到的情緒為：{dominant_emotion}**")

        # 對應提示語生成客服語句
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
