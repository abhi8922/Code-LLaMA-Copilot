import os
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set cache directory to avoid repeated downloads
os.environ["HF_HOME"] = "/tmp/huggingface"

# Model name (use smaller model for better performance)
MODEL_NAME = "codellama/CodeLlama-3b-Instruct-hf"  # Change to "7b" if needed

@st.cache_resource()
def load_model():
    """Load the Code LLaMA model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32,  # Use CPU-friendly float32
        device_map="cpu"  # Force CPU usage
    )
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

def generate_code(prompt, max_length=512):
    """Generate code suggestions based on user input."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(
        **inputs, 
        max_length=max_length, 
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
def main():
    st.title("Code LLaMA Copilot")
    st.write("Enter a programming prompt below to get AI-generated code suggestions.")

    user_prompt = st.text_area("Your Code Prompt:", "")
    if st.button("Generate Code"):
        if user_prompt:
            st.write("Generating code...")
            result = generate_code(user_prompt)
            st.code(result, language="python")
        else:
            st.warning("Please enter a prompt!")

if __name__ == "__main__":
    main()
