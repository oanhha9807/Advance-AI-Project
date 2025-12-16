from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import os
import torch
import requests
import uvicorn
import nest_asyncio
import streamlit as st
# from pyngrok import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms.base import LLM
from peft import PeftModel, PeftConfig
from pinecone import Pinecone, ServerlessSpec
from typing import Optional, List, Mapping, Any
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
login("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")

os.environ['PINECONE_API_KEY'] = "pcsk_5dMf2h_5hyhbhRGp6DhQR95dUbtJGLkYoc69cqYmgUwXjyuy4Ghh5cBhLap47BC29e3ESb"
os.environ['HF_API_KEY'] = "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"
# Print CUDA availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU being used:", torch.cuda.get_device_name(0))
base_model_id = "meta-llama/Llama-2-7b-chat-hf"
adapter_model_id = "ijuliet/Llama-2-7b-chat-hf-mental-health"

# base_model_id = "TinyPixel/Llama-2-7B-bf16-sharded"
# adapter_model_id = "TinyPixel/Llama-2-7B-bf16-sharded"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,

)
# Load the PEFT adapter
peft_config = PeftConfig.from_pretrained(adapter_model_id, use_auth_token = "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" )
model_to_merge = PeftModel.from_pretrained(model, adapter_model_id, use_auth_token = "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" )

# Merge the base model with the adapter
model_to_merge.merge_and_unload()
model_to_merge.push_to_hub("ijuliet/llama-2-mental-health-merged")
tokenizer.push_to_hub("ijuliet/llama-2-mental-health-merged")
# Load merged model with 8-bit quantization
model_id = "ijuliet/llama-2-mental-health-merged"
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("\nLoading model with 8-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,  # Use 8-bit quantization, Converts model weights from 32-bit to 8-bit integers
    torch_dtype=torch.float16, # # Uses half-precision floating point for activations
    low_cpu_mem_usage=True
)

# Test model
test_prompt = """You are a compassionate emotional support companion. Provide a complete, empathetic, non-judgmental, thoughtful response.

Question: I have a social event coming up soon, and I'm feeling anxious. Can you suggest ways to overcome this?
Answer:"""

print("\nTokenizing input...")
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

print("\nGenerating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7, #controlled randomness to avoid repetitive responses
        top_p=0.9, # only consider tokens whose cumulative probability reaches 90%
        top_k=50, # only consider the 50 most likely next tokens
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nResponse:", response)

