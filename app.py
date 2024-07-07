import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from src.utils import load_config
from src.logger import setup_logger

logger = setup_logger(__name__)

@st.cache_resource
def load_fine_tuned_model():
    config = load_config()
    
    logger.info("Loading fine-tuned model")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, config['model']['new_model'])
    model = model.merge_and_unload()

    logger.info("Fine-tuned model loaded successfully")
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']

def main():
    st.set_page_config(page_title="Bhagwad Gita Chat", page_icon="🕉️")
    st.title("Bhagwad Gita Chat")

    # Load the fine-tuned model
    model, tokenizer = load_fine_tuned_model()

    # User input
    user_input = st.text_input("Ask a question about the Bhagwad Gita:", key="user_input")

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
    st.markdown("This chatbot uses a fine-tuned LLaMA model trained on the Bhagwad Gita dataset.")

if __name__ == "__main__":
    main()