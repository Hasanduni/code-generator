from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device("cpu")

# Use a smaller GPT-2 model for testing
model_name = "gpt2"

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    return model, tokenizer
