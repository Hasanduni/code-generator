import streamlit as st
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Set device to CPU to avoid GPU issues
device = torch.device("cpu")

# Initialize and load the smaller version of LLaMA (7B)
@st.cache_resource
def load_model():
    model_name = "huggingface/llama-7b"  # LLaMA 7B for lighter version
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app UI setup
st.title("Lighter LLaMA Content Generation")
st.write("Enter a prompt and generate content using a lighter version of LLaMA!")

# User input for content generation
prompt = st.text_area("Enter your prompt:", height=150)

if st.button("Generate Content"):
    if prompt:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate output from LLaMA
        outputs = model.generate(**inputs, max_length=200)
        
        # Decode and display the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(generated_text)
    else:
        st.write("Please enter a prompt to generate content.")
