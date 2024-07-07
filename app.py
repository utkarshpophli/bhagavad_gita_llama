import os
import numpy
import tempfile
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from src.utils import load_config
from src.logger import setup_logger

logger = setup_logger(__name__)

@st.cache_resource
def load_fine_tuned_model():
    logger.info("Loading model")
    try:
        tokenizer = AutoTokenizer.from_pretrained("utkarshpophli/bhagavad_gita_llama", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(
            "utkarshpophli/bhagavad_gita_llama",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load the model. Error: {str(e)}")
        return None, None

def generate_response(prompt, model, tokenizer):
    if model is None or tokenizer is None:
        return "Sorry, the model is not available. Please try again later."
    
    try:
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
        result = pipe(f"<s>[INST] {prompt} [/INST]")
        return result[0]['generated_text']
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"An error occurred while generating the response: {str(e)}"

def main():
    st.set_page_config(page_title="Bhagavad Gita Chat", page_icon="🕉️")
    st.title("Bhagavad Gita Chat")

    # Load the model
    model, tokenizer = load_fine_tuned_model()

    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please try again later.")
        return

    # User input
    user_input = st.text_input("Ask a question about the Bhagavad Gita:", key="user_input")

    if st.button("Generate Response"):
        if user_input:
            logger.info(f"User input: {user_input}")
            with st.spinner("Generating response..."):
                response = generate_response(user_input, model, tokenizer)
            
            # Extract the response after the input prompt
            response_start = response.find("[/INST]") + 7
            cleaned_response = response[response_start:].strip()
            
            st.markdown("### Response:")
            st.write(cleaned_response)
            logger.info(f"Generated response: {cleaned_response}")
        else:
            st.warning("Please enter a question.")

    st.markdown("---")
    st.markdown("This chatbot uses a LLaMA model fine-tuned on the Bhagavad Gita dataset.")

if __name__ == "__main__":
    main()