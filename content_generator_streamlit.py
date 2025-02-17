import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Function to generate content based on the prompt
def generate_content(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_length=max_length, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit interface
st.title("Content Generator with TinyLlama")
prompt = st.text_input("Enter Prompt")
max_length = st.slider("Max Length", min_value=50, max_value=500, value=150)
if prompt:
    generated_text = generate_content(prompt, max_length)
    st.write(generated_text)
