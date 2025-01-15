from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Use a lighter model (TinyLLaMA or LLaMA-2 3B)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Function to generate content based on the prompt
def generate_content(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, max_length=max_length, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for content generation
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    max_length = int(request.form['max_length'])
    
    generated_text = generate_content(prompt, max_length)
    
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
