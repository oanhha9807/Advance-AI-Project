import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# TokyoNight color scheme
TOKYONIGHT = {
    "background": "#1a1b26",
    "foreground": "#a9b1d6",
    "primary": "#7aa2f7",
    "secondary": "#9ece6a",
    "accent": "#bb9af7",
    "warning": "#ff9e64",
    "error": "#f7768e"
}

# Set up the page with TokyoNight theme
st.set_page_config(
    page_title="Feel Emotion☺️😣😡- Emotional AI Assistant",
    page_icon="😊",
    layout="wide"
)

# Apply custom CSS
def apply_tokyonight():
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {TOKYONIGHT['background']};
            color: {TOKYONIGHT['foreground']};
        }}
        .stChatInput textarea {{
            background-color: {TOKYONIGHT['background']} !important;
            color: {TOKYONIGHT['foreground']} !important;
            border-color: {TOKYONIGHT['primary']} !important;
        }}
        .stChatInput button {{
            background-color: {TOKYONIGHT['primary']} !important;
            color: {TOKYONIGHT['background']} !important;
        }}
        .stChatMessage {{
            border-left: 3px solid {TOKYONIGHT['primary']};
        }}
        .user-message {{
            background-color: {TOKYONIGHT['background']};
            border-color: {TOKYONIGHT['secondary']} !important;
        }}
        .assistant-message {{
            background-color: #24283b;
            border-color: {TOKYONIGHT['primary']} !important;
        }}
        h1 {{
            color: {TOKYONIGHT['primary']} !important;
        }}
        .caption {{
            color: {TOKYONIGHT['secondary']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_tokyonight()

# Title and caption
st.title("Feel Emotion☺️- Your Emotional Support Companion")
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
    st.title("😊 Settings")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        save_history([])
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"""
    <div style="color:{TOKYONIGHT['secondary']}">
    <small>Mohammed is here to:</small>
    <ul>
        <li>Listen without judgment</li>
        <li>Validate your feelings</li>
        <li>Offer gentle support</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)