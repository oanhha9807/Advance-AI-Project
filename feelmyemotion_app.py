
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
from datetime import datetime
from deepface import DeepFace
from PIL import Image, ImageDraw
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tempfile
import plotly.graph_objects as go
import cv2
import base64
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
import time
from streamlit_extras.stylable_container import stylable_container
import altair as alt

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Set page config
st.set_page_config(page_title="FEEL MY EMOTION☺️😨😡", layout="wide", page_icon="😊")

# --- Constants ---
EMOTION_GIFS = {
    "happy": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/Happy.gif",
    "angry": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/gif/angry.gif",
    "neutral": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/Neutral.gif",
    "sad": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/Sad.gif",
    "disgust": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/gif/disgust.gif",
    "fear": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/Disgust.gif",
    "pleasant surprise": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/gif/surprise.gif",
    "love": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/gif/love.gif",
    "humor": "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition%20_App_Using_Streamlit/images/Emoji/gif/humor.gif"
}

# Background image URL
BACKGROUND_IMAGE = "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition _App_Using_Streamlit/images/backgound.gif"


# --- Helper Functions ---
def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{BACKGROUND_IMAGE}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_models():
    # Load text model
    text_model_url = "https://github.com/ADHIL48/Emotion_Recognition_app/raw/f1096452ed11fde1454dbd13d6e006a9a8ea1412/text/models/emotion_classifier_pipe_lr.pkl"
    text_model_path = "./emotion_classifier_pipe_lr.pkl"
    if not os.path.exists(text_model_path):
        urllib.request.urlretrieve(text_model_url, text_model_path)
    pipe_lr = joblib.load(open(text_model_path, "rb"))
    
    # Load speech model
    cnn_model_url = "https://raw.githubusercontent.com/ADHIL48/Emotion_Recognition_app/main/speech_or_voice/models/CnnModel.h5"
    response = requests.get(cnn_model_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(response.content)
        cnn_model = load_model(tmp.name)
    
    return pipe_lr, cnn_model

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def analyze_faces(img_array):
    try:
        results = DeepFace.analyze(img_array, actions=['emotion', 'gender'], enforce_detection=False)
        
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)
        
        for result in results:
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
            draw.rectangle([(x, y), (x+w, y+h)], outline="red", width=2)
            
            emotion = result['dominant_emotion']
            gender = result['dominant_gender']
            emotion_score = result['emotion'][emotion]
            gender_score = result['gender'][gender]
            
            # Add emoji based on emotion
            emoji_dict = {
                "happy": "😊",
                "angry": "😠",
                "neutral": "😐",
                "sad": "😢",
                "disgust": "🤢",
                "fear": "😨",
                "pleasant surprise": "😲",
                "love": "❤",
                "humor": "😂"
            }
            emoji = emoji_dict.get(emotion.lower(), "❓")
            
            text = f"{emoji} {emotion} ({emotion_score:.1f}%)\n{gender} ({gender_score:.1f}%)"
            draw.text((x, y-50), text, fill="red")
        
        return img_pil, results
    except Exception as e:
        st.error(f"Face analysis error: {str(e)}")
        return None, None

def predict_speech_emotion(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        data, sampling_rate = librosa.load(tmp.name)
    
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    X_test = np.expand_dims([mfccs], axis=2)
    
    predict = st.session_state.cnn_model.predict(X_test)
    predictions = np.argmax(predict, axis=1)
    emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "pleasant surprise"]
    detected_emotion = [emotion_labels[val] for val in predictions][0]
    
    return detected_emotion, predict[0]

def wave_plot(data, sampling_rate):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 4)
    ax.set_facecolor("black")
    fig.set_facecolor("black")
    plt.ylabel('Amplitude')
    plt.title("WAVEFORM", fontweight="bold")
    librosa.display.waveshow(data, sr=sampling_rate, x_axis='s')
    st.pyplot(fig, use_container_width=True)
    return data

def plot_radar_chart(emotion_probs, title):
    categories = list(emotion_probs.keys())
    values = list(emotion_probs.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def get_wellbeing_recommendation(emotion):
    recommendations = {
        "happy": "Great! Consider journaling about what's making you happy",
        "sad": "Try calling a friend or going for a walk",
        "angry": "Take deep breaths and count to 10",
        "neutral": "Practice mindfulness to connect with your emotions",
        "fear": "Try grounding techniques like the 5-4-3-2-1 method",
        "disgust": "Reflect on what triggered this reaction",
        "pleasant surprise": "Enjoy this positive moment",
        "love": "Express your feelings to the person you care about",
        "humor": "Share the laughter with others"
    }
    return recommendations.get(emotion.lower(), "Try mindfulness meditation")

def display_emotion_results(emotion, probs, modality, content=None):
    st.success(f"Detected Emotion: {emotion}")
    st.image(EMOTION_GIFS.get(emotion.lower(), ""), width=200)
    
    if modality == "text":
        emotion_probs = dict(zip(st.session_state.pipe_lr.classes_, probs*100))
    elif modality == "face":
        emotion_probs = probs
    else:  # speech
        emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "pleasant surprise"]
        emotion_probs = dict(zip(emotion_labels, probs*100))
    
    plot_radar_chart(emotion_probs, f"{modality.capitalize()} Emotion Probabilities")
    
    proba_df = pd.DataFrame([emotion_probs]).T.reset_index()
    proba_df.columns = ["emotions", "probability"]
    
    fig = alt.Chart(proba_df).mark_bar().encode(
        x='emotions',
        y='probability',
        color='emotions'
    )
    st.altair_chart(fig, use_container_width=True)
    
    st.subheader("💡 Recommendation")
    st.write(get_wellbeing_recommendation(emotion))
    
    # Store in history
    history_entry = {
        "timestamp": datetime.now(),
        "modality": modality,
        "emotion": emotion,
        "confidence": max(probs) if modality != "face" else probs[emotion],
    }
    
    if content:
        history_entry["content"] = content[:50] + "..."
    
    st.session_state.history.append(history_entry)
    
    # Update streak if it's a new day
    last_date = st.session_state.get('last_analysis_date')
    today = datetime.now().date()
    if last_date != today:
        st.session_state.streak += 1
        st.session_state.last_analysis_date = today

def feature_card(title, description, icon):
    with stylable_container(
        key=f"feature_{title}",
        css_styles="""
        {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            height: 200px;
            margin: 10px;
            color: white;
        }
        """
    ):
        st.subheader(f"{icon} {title}")
        st.write(description)

def chat_with_ai(prompt):
    try:
        # Using Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error in AI chat: {str(e)}")
        return "I'm sorry, I couldn't process your request right now. Please try again later."

def send_feedback_email(name, email, rating, comments):
    try:
        # Email configuration
        sender_email = "mohammedadhil0408@gmail.com"
        receiver_email = "mohammedadhil0408@gmail.com"
        password = "your_app_password_here"  # Use app password for security
        
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = "New Feedback from Emotion Recognition App"
        
        # Email body
        body = f"""
        Name: {name}
        Email: {email}
        Rating: {rating} stars
        Comments: {comments}
        """
        
        message.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# --- Initialize Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []
    
if 'streak' not in st.session_state:
    st.session_state.streak = 0

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'last_analysis_date' not in st.session_state:
    st.session_state.last_analysis_date = None

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# --- Load Models ---
pipe_lr, cnn_model = load_models()
st.session_state.pipe_lr = pipe_lr
st.session_state.cnn_model = cnn_model

# Set background
set_background()


# --- UI Setup ---
# Sidebar Navigation
with st.sidebar:
    # Set sidebar background and text color
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-image: url("https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition _App_Using_Streamlit/images/background_menu.png");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        
        /* Make all sidebar text black */
        .sidebar .stSelectbox, 
        .sidebar .stTitle, 
        .sidebar .stSubheader,
        .sidebar .stMarkdown,
        .sidebar .stMetric {
            color: black !important;
        }
        
        /* Transparent containers with black text */
        .sidebar .stMarkdown div {
            color: black !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Add logo at the top with some padding
    st.markdown('<div style="padding: 20px 0;">', unsafe_allow_html=True)
    logo_url = "https://github.com/ADHIL48/Emotion_Recognition_app/raw/main/Emotion_Recognition _App_Using_Streamlit/images/logo.png"
    st.image(logo_url, width=150)
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    
    st.markdown("<h1 style='color: black;'>Menu</h1>", unsafe_allow_html=True)
    app_mode = st.selectbox("Navigatio", [
        "Home", 
        "Text Emotion Recognition", 
        "Face Emotion Recognition", 
        "Speech Emotion Recognition",
        "Multi-Modal Emotion Recognition",
        "AI Emotional Support Chat",
        "History Dashboard",
        "User Manual",
        "About the App"
    ])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
   
    


if app_mode == "Home":
    # Create a header container
    header = st.container()
    
    with header:
        # Create columns for layout (3 columns to center the title)
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Empty column for balance
            
            pass
            
        with col2:
            # Centered title and logo
            st.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 10px;">
                <img src="{logo_url}" width="60">
                <h1 style="color: white; margin: 0; text-align: center;">FEEL MY EMOTION</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            # Lottie animation in top-right corner - 2.5 inches ≈ 240 pixels
            st_lottie(load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_1pxqjqps.json"),
                      speed=1, 
                      height=200,  # Approximately 2.5 inches
                      key="home")
    # Features Grid
    with st.container():
        cols = st.columns(3)
        with cols[0]:
            feature_card("Text Emotion Recognition", "Analyze emotions in any text content", "📝")
        with cols[1]:
            feature_card("Face Emotion Recognition", "Real-time multi-face emotion detection", "📷")
        with cols[2]:
            feature_card("Speech Emotion Recognition", "Record and analyze speech emotions", "🎤")
        with cols[0]:
            feature_card("Multi-Modal Analysis", "Combine text, face and voice for better accuracy", "🎭")
        with cols[1]:
            feature_card("AI Emotional Support", "Chat with our AI assistant for guidance", "💬")
        with cols[2]:
            feature_card("History Dashboard", "Track your emotional patterns over time", "📊")
    
    # About the App
    with st.container():
        with stylable_container(
            key="about_card",
            css_styles="""
            {
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border: 2px solid #4CAF50;
                color: white;
            }
            """
        ):
            st.subheader("About the App")
            st.write("The Emotion Recognition app leverages state-of-the-art technologies to analyze and detect emotions from facial expressions, speech, and text. It aims to help users understand their emotional states in real-time, providing a deeper connection with their mental well-being.")
    
    # Vision and Mission
    with st.container():
        with stylable_container(
            key="vision_card",
            css_styles="""
            {
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border: 2px solid #2196F3;
                color: white;
            }
            """
        ):
            st.subheader("Vision and Mission")
            st.write("Our vision is to revolutionize emotional intelligence through technology. The mission is to create an easy-to-use platform that can detect emotions using facial expressions and speech patterns, helping users gain insights into their emotional states.")
    
    # Key Features and Technology Used (Two columns)
    with st.container():
        cols = st.columns(2)
        with cols[0]:
            with stylable_container(
                key="features_card",
                css_styles="""
                {
                    background-color: rgba(0, 0, 0, 0.7);
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border: 2px solid #FF5722;
                    height: 100%;
                    color: white;
                }
                """
            ):
                st.subheader("Key Features")
                st.write("Real-time Emotion Recognition")
                st.write("Emotion-Based GIFs")
                st.write("Multiple Input Modes")
                st.write("Personalized Feedback")
                st.write("Multi-Face Detection")
                st.write("Emotional Support Chat")
        
        with cols[1]:
            with stylable_container(
                key="tech_card",
                css_styles="""
                {
                    background-color: rgba(0, 0, 0, 0.7);
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border: 2px solid #9C27B0;
                    height: 100%;
                    color: white;
                }
                """
            ):
                st.subheader("Technology Used")
                st.write("DeepFace for facial emotion recognition")
                st.write("Librosa for audio processing")
                st.write("Scikit-learn for text emotion classification")
                st.write("Streamlit for interactive UI")
                st.write("Google Gemini for AI chat support")
    
    # Developed By
    with st.container():
        with stylable_container(
            key="dev_card",
            css_styles="""
            {
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border: 2px solid #FFC107;
                color: white;
            }
            """
        ):
            st.subheader("Developed By")
            st.write("ADHIL M")
            st.write("Contact: mohammedadhil0408@gmail.com")
            st.write("GitHub: github/adhil48")
    
    # Emotion Gallery
    st.subheader("Emotion Gallery")
    emotion_cols = st.columns(7)
    emotions = list(EMOTION_GIFS.keys())
    for i, col in enumerate(emotion_cols):
        if i < len(emotions):
            with col:
                st.image(EMOTION_GIFS[emotions[i]], width=100, caption=emotions[i].capitalize())

elif app_mode == "Text Emotion Recognition":
    st.title("📝 Text Emotion Recognition")
    st.write("Analyze emotions from text input or uploaded documents")
    
    # Input options with tabs
    tab1, tab2 = st.tabs(["Type Text", "Upload Document"])
    
    with tab1:
        # Initialize session state if not exists
        if 'text_input' not in st.session_state:
            st.session_state.text_input = ""
        if 'text_results' not in st.session_state:
            st.session_state.text_results = None
        
        # Create the text area widget
        user_text = st.text_area(
            "Enter your text here", 
            height=150, 
            key="text_area_widget",
            value=st.session_state.text_input
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analyze Emotion", key="analyze_text"):
                if user_text.strip():
                    st.session_state.text_input = user_text  # Store the current text
                    pred = st.session_state.pipe_lr.predict([user_text])[0]
                    proba = st.session_state.pipe_lr.predict_proba([user_text])[0]
                    st.session_state.text_results = (pred, proba, user_text)
                else:
                    st.warning("Please enter some text to analyze")
        
        with col2:
            if st.button("Clear", key="clear_text"):
                st.session_state.text_input = ""  # Clear the stored text
                st.session_state.text_results = None  # Clear the results
                st.rerun()  # Rerun to update the widget
        
        # Display results if they exist
        if st.session_state.text_results:
            pred, proba, text_content = st.session_state.text_results
            display_emotion_results(pred, proba, "text", text_content)
    
    with tab2:
        uploaded_file = st.file_uploader("Upload a document (TXT)", type=["txt"], key="doc_upload")
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            st.text_area("Extracted Text", text, height=150, key="extracted_text")
            
            if st.button("Analyze Document", key="analyze_doc"):
                pred = st.session_state.pipe_lr.predict([text])[0]
                proba = st.session_state.pipe_lr.predict_proba([text])[0]
                display_emotion_results(pred, proba, "text", text)

elif app_mode == "Face Emotion Recognition":
    st.title("📷 Face Emotion Recognition")
    st.write("Detect emotions from photos or live camera input with multi-face support")
    
    # Input options with tabs
    tab1, tab2 = st.tabs(["Take Photo", "Take Video"])
    
    with tab1:
        img_file_buffer = st.camera_input("Take a photo for emotion analysis", key="camera_input")
        
        if img_file_buffer is not None:
            img = Image.open(img_file_buffer)
            img_array = np.array(img.convert('RGB'))
            
            annotated_img, results = analyze_faces(img_array)
            
            if annotated_img and results:
                st.image(annotated_img, use_container_width=True)
                st.success(f"Detected {len(results)} face(s)")
                
                for i, result in enumerate(results):
                    with st.expander(f"Face {i+1} Details"):
                        dominant_emotion = result['dominant_emotion']
                        display_emotion_results(dominant_emotion, result['emotion'], "face")
        
        if st.button("Clear", key="clear_face"):
            st.session_state.camera_input = None
            st.rerun()
    
    with tab2:
        st.write("Real-time Video Emotion Detection")
        run = st.checkbox('Start Webcam', key='video_run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        # Placeholder for analysis results
        results_placeholder = st.empty()
        
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze frame
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # Display results
                with results_placeholder.container():
                    if results:
                        dominant_emotion = results[0]['dominant_emotion']
                        emotion_probs = results[0]['emotion']
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.image(EMOTION_GIFS.get(dominant_emotion.lower(), ""), width=200)
                            st.success(f"Detected Emotion: {dominant_emotion}")
                        
                        with cols[1]:
                            plot_radar_chart(emotion_probs, "Emotion Probabilities")
                            
                            proba_df = pd.DataFrame([emotion_probs]).T.reset_index()
                            proba_df.columns = ["emotions", "probability"]
                            st.altair_chart(alt.Chart(proba_df).mark_bar().encode(
                                x='emotions',
                                y='probability',
                                color='emotions'
                            ), use_container_width=True)
                
                # Draw on frame
                for result in results:
                    x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    emotion = result['dominant_emotion']
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
            
            FRAME_WINDOW.image(frame)
        
        camera.release()
        results_placeholder.empty()

elif app_mode == "Speech Emotion Recognition":
    st.title("🎤 Speech Emotion Recognition")
    st.write("Analyze emotions from live recordings or uploaded audio files")
    
    # Input options with tabs
    tab1, tab2 = st.tabs(["Record Live", "Upload Audio"])
    
    with tab1:
        st.write("Click to record (5 seconds max):")
        audio_bytes = audio_recorder(pause_threshold=5.0, key="audio_recorder")
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            # Save to temp file to plot waveform
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            try:
                data, sampling_rate = librosa.load(tmp_path)
                st.markdown("### Audio Waveform")
                wave_plot(data, sampling_rate)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Analyze Recording", key="analyze_recording"):
                        try:
                            emotion, probs = predict_speech_emotion(audio_bytes)
                            display_emotion_results(emotion, probs, "speech")
                        except Exception as e:
                            st.error(f"Error analyzing audio: {str(e)}")
                
                with col2:
                    if st.button("Clear", key="clear_audio"):
                        st.rerun()
            finally:
                os.unlink(tmp_path)
    
    with tab2:
        audio_file = st.file_uploader("Upload audio file", type=['wav'], key="audio_upload")
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            
            try:
                data, sampling_rate = librosa.load(tmp_path)
                st.audio(tmp_path)
                
                st.markdown("### Audio Waveform")
                wave_plot(data, sampling_rate)
                
                with open(tmp_path, "rb") as f:
                    audio_bytes = f.read()
                
                emotion, probs = predict_speech_emotion(audio_bytes)
                display_emotion_results(emotion, probs, "speech")
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
            finally:
                os.unlink(tmp_path)

elif app_mode == "Multi-Modal Emotion Recognition":
    st.title("🎭 Multi-Modal Emotion Recognition")
    st.write("Combine inputs from text, facial expressions, and speech for more accurate emotion detection")
    
    if 'multi_modal_stage' not in st.session_state:
        st.session_state.multi_modal_stage = 0
        st.session_state.multi_modal_data = {}
    
    if st.session_state.multi_modal_stage == 0:
        st.subheader("Step 1: Text Analysis")
        user_text = st.text_area("Enter text for analysis", height=100, key="multi_text")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Next", key="text_next"):
                if user_text.strip():
                    pred = st.session_state.pipe_lr.predict([user_text])[0]
                    proba = st.session_state.pipe_lr.predict_proba([user_text])[0]
                    st.session_state.multi_modal_data['text'] = {
                        'emotion': pred,
                        'confidence': max(proba),
                        'proba': dict(zip(st.session_state.pipe_lr.classes_, proba)),
                        'content': user_text
                    }
                    st.session_state.multi_modal_stage = 1
                    st.rerun()
                else:
                    st.warning("Please enter some text")
        
        with col2:
            if st.button("Clear", key="text_clear"):
                st.session_state.multi_text = ""
                st.session_state.multi_modal_data = {}
                st.rerun()
    
    elif st.session_state.multi_modal_stage == 1:
        st.subheader("Step 2: Face Analysis")
        img_file = st.camera_input("Take a facial photo", key="multi_face")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Next", key="face_next"):
                if img_file:
                    img_array = np.array(Image.open(img_file).convert('RGB'))
                    annotated_img, face_results = analyze_faces(img_array)
                    
                    if face_results:
                        st.image(annotated_img, use_container_width=True)
                        dominant = face_results[0]['dominant_emotion']
                        conf = face_results[0]['emotion'][dominant]
                        st.session_state.multi_modal_data['face'] = {
                            'emotion': dominant,
                            'confidence': conf,
                            'proba': face_results[0]['emotion'],
                            'annotated_img': annotated_img
                        }
                        st.session_state.multi_modal_stage = 2
                        st.rerun()
                else:
                    st.warning("Please take a photo")
        
        with col2:
            if st.button("Back", key="face_back"):
                st.session_state.multi_modal_stage = 0
                st.rerun()
    
    elif st.session_state.multi_modal_stage == 2:
        st.subheader("Step 3: Voice Analysis")
        audio_bytes = audio_recorder(key="multi_voice")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Analyze", key="voice_analyze"):
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Save to temp file to plot waveform
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    try:
                        data, sampling_rate = librosa.load(tmp_path)
                        st.markdown("### Audio Waveform")
                        wave_plot(data, sampling_rate)
                        
                        emotion, probs = predict_speech_emotion(audio_bytes)
                        emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "pleasant surprise"]
                        st.session_state.multi_modal_data['speech'] = {
                            'emotion': emotion,
                            'confidence': max(probs),
                            'proba': dict(zip(emotion_labels, probs)),
                            'waveform': (data, sampling_rate),
                            'audio_bytes': audio_bytes
                        }
                        st.session_state.multi_modal_stage = 3
                        st.rerun()
                    finally:
                        os.unlink(tmp_path)
                else:
                    st.warning("Please record some audio")
        
        with col2:
            if st.button("Back", key="voice_back"):
                st.session_state.multi_modal_stage = 1
                st.rerun()
    
    elif st.session_state.multi_modal_stage == 3:
        st.subheader("Combined Analysis Results")
        
        # Display all collected data
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("### Text Analysis")
            text_data = st.session_state.multi_modal_data['text']
            st.write(f"Emotion: {text_data['emotion']}")
            st.image(EMOTION_GIFS.get(text_data['emotion'].lower(), ""), width=150)
            st.write(f"Confidence: {text_data['confidence']*100:.1f}%")
            
            st.markdown("### Face Analysis")
            face_data = st.session_state.multi_modal_data['face']
            st.image(face_data['annotated_img'], use_container_width=True)
            st.write(f"Emotion: {face_data['emotion']}")
            st.image(EMOTION_GIFS.get(face_data['emotion'].lower(), ""), width=150)
            st.write(f"Confidence: {face_data['confidence']:.1f}%")
        
        with cols[1]:
            st.markdown("### Voice Analysis")
            speech_data = st.session_state.multi_modal_data['speech']
            st.audio(speech_data['audio_bytes'], format="audio/wav")
            wave_plot(*speech_data['waveform'])
            st.write(f"Emotion: {speech_data['emotion']}")
            st.image(EMOTION_GIFS.get(speech_data['emotion'].lower(), ""), width=150)
            st.write(f"Confidence: {speech_data['confidence']*100:.1f}%")
        
        # Combine results - take the result with highest confidence
        best_modality = max(st.session_state.multi_modal_data.items(), 
                          key=lambda x: x[1]['confidence'])
        final_emotion = best_modality[1]['emotion']
        
        st.success(f"Final Detected Emotion: {final_emotion} (from {best_modality[0]} analysis)")
        st.image(EMOTION_GIFS.get(final_emotion.lower(), ""), width=200)
        
        # Show radar chart combining all probabilities
        combined_probs = {}
        for modality, data in st.session_state.multi_modal_data.items():
            if isinstance(data['proba'], dict):
                for emotion, prob in data['proba'].items():
                    if emotion in combined_probs:
                        combined_probs[emotion] += prob * data['confidence']
                    else:
                        combined_probs[emotion] = prob * data['confidence']
        
        # Normalize probabilities
        total = sum(combined_probs.values())
        combined_probs = {k: v/total*100 for k, v in combined_probs.items()}
        plot_radar_chart(combined_probs, "Combined Emotion Probabilities")
        
        # Store in history
        st.session_state.history.append({
            "timestamp": datetime.now(),
            "modality": "multi-modal",
            "emotion": final_emotion,
            "confidence": best_modality[1]['confidence'],
            "details": str(st.session_state.multi_modal_data)
        })
        
        # Show recommendation
        st.subheader("💡 Recommendation")
        st.write(get_wellbeing_recommendation(final_emotion))
        
        # Update streak
        last_date = st.session_state.get('last_analysis_date')
        today = datetime.now().date()
        if last_date != today:
            st.session_state.streak += 1
            st.session_state.last_analysis_date = today
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start New Analysis", key="new_analysis"):
                st.session_state.multi_modal_stage = 0
                st.session_state.multi_modal_data = {}
                st.rerun()
        with col2:
            if st.button("Clear All", key="clear_all"):
                st.session_state.multi_modal_stage = 0
                st.session_state.multi_modal_data = {}
                st.session_state.multi_text = ""
                st.session_state.multi_face = None
                st.rerun()


elif app_mode == "AI Emotional Support Chat":
    # Title and caption
    st.title("Feel Emotion☺️- AI Emotional Support Companion")
    st.caption("A compassionate AI here to listen and understand your feelings")

    # Initialize chat history with persistence
    HISTORY_FILE = "chat_history.json"

    def load_history():
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return []

    def save_history(messages):
        with open(HISTORY_FILE, "w") as f:
            json.dump(messages, f)

    if "messages" not in st.session_state:
        st.session_state.messages = load_history()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    # System prompt for emotional support
    SYSTEM_PROMPT = """Your name is Mohammed. You are a warm, empathetic emotional support assistant.
    Respond with compassion and understanding. Validate feelings before offering suggestions.
    Keep responses concise but meaningful (1-3 sentences max). 
    Remove any <think> tags from your responses.
    Use a gentle, caring tone with occasional emojis (🌙✨💭).
    Focus on active listening and emotional validation."""

    # Clean response function
    def clean_response(response):
        return response.replace("<think>", "").replace("</think>", "").strip()

    # Set up the Ollama model
    def setup_llm():
        llm = Ollama(
            model="deepseek-r1:1.5b",
            temperature=0.8,  # Slightly more creative responses
            top_k=40,
            repeat_penalty=1.1
        )
        
        # Convert message history to proper message types
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in st.session_state.messages[-6:]:  # Keep last 6 messages for context
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        prompt = ChatPromptTemplate.from_messages(messages)
        
        return prompt | llm | StrOutputParser()

    # Main chat interface
    if prompt := st.chat_input("Share your thoughts..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="😊"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Initialize the LLM chain
            chain = setup_llm()
            
            # Stream the response
            for chunk in chain.stream({"input": prompt}):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            # Clean and display final response
            cleaned_response = clean_response(full_response)
            message_placeholder.markdown(cleaned_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
        
        # Persist history
        save_history(st.session_state.messages)

    # Add sidebar with controls
    with st.sidebar:
       
        if st.button("Clear Chat History", type="primary"):
            st.session_state.messages = []
            save_history([])
            st.rerun()
      



elif app_mode == "History Dashboard":
    st.title("📊 Emotion History Dashboard")
    
    if not st.session_state.history:
        st.warning("No history yet. Analyze some emotions first!")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.history)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Show raw data
        st.dataframe(df.sort_values('timestamp', ascending=False))
        
        # Visualizations
        st.subheader("📈 Emotion Trends")
        tab1, tab2, tab3 = st.tabs(["Timeline", "Distribution", "Modality Comparison"])
        
        with tab1:
            st.line_chart(df.set_index('timestamp')['emotion'].value_counts())
        
        with tab2:
            st.bar_chart(df['emotion'].value_counts())
        
        with tab3:
            st.bar_chart(pd.crosstab(df['modality'], df['emotion']))
        
        # Export options
        st.subheader("📤 Export Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download as CSV",
            csv,
            "emotion_history.csv",
            "text/csv"
        )

elif app_mode == "User Manual":
    st.title("📖 User Manual")
    
    with st.expander("How to use Text Emotion Recognition"):
        st.write("""
        1. Type or paste your text in the input box
        2. Click 'Analyze Emotion' button
        3. View the detected emotion and recommendations
        """)
    
    with st.expander("How to use Face Emotion Recognition"):
        st.write("""
        1. Take a photo using your camera
        2. The app will detect faces and emotions
        3. View detailed analysis for each face
        """)
    
    with st.expander("How to use Real-time Video Analysis"):
        st.write("""
        1. Go to Face Emotion Recognition tab
        2. Select 'Take Video' option
        3. Click 'Start Webcam' to begin
        4. View live emotion detection results
        """)
    
    with st.expander("How to use Speech Emotion Recognition"):
        st.write("""
        1. Record your voice (5 seconds max)
        2. Click 'Analyze Recording'
        3. View the emotion analysis results
        """)
    
    with st.expander("How to use Multi-Modal Emotion Recognition"):
        st.write("""
        1. Enter text for analysis and click Next
        2. Take a photo of your face and click Next
        3. Record your voice and click Analyze
        4. View combined results and recommendations
        """)

elif app_mode == "About the App":
    st.title("About the App☺️")
    
    st.write("""
    This Emotion Recognition App was developed to help people understand and manage their emotional states.
    It combines multiple modalities (text, face, and speech) to provide comprehensive emotional analysis.
    """)
    
    st.write("Version: 1.0")
    st.write("Last Updated: April 2025")
    st.write("Developed By : Adhil M")
    st.write("Contact: mohammedadhil0408@gmail.com")
    st.write("GitHub: github/adhil48")

    st.subheader("Inspiration")
    st.write("""
    The app was inspired by the growing need for emotional awareness and mental well-being tools.
    By combining AI technologies with user-friendly interfaces, we aim to make emotional intelligence
    accessible to everyone.
    """)

