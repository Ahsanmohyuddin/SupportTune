
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "/content/drive/MyDrive/SupportTune/models/qwen25-05b-qlora-r16"

SYSTEM_PROMPT = """
You are SupportTune, an experimental customer-support assistant.
Be clear, polite, relevant, and concise.
Do not claim to access accounts, check live orders, issue refunds,
change orders, or perform other actions unless an actual tool provides
that capability. Never invent company policies, order details, or dates.
If information is missing, ask a relevant clarifying question.
"""

st.set_page_config(page_title="SupportTune", page_icon="💬")
st.title("💬 SupportTune")
st.caption("Experimental customer-support assistant fine-tuned with QLoRA")

st.warning(
    "Demo only: this model has no live access to customer accounts, "
    "orders, payments, or company systems. Do not enter personal or "
    "sensitive information."
)

@st.cache_resource
def load_model():
    # Load the original model in 4-bit to reduce GPU memory use.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )

    # Load the trained adapter without modifying the original model files.
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return tokenizer, model

try:
    with st.spinner("Loading SupportTune model..."):
        tokenizer, model = load_model()
    st.success("Model loaded.")
except Exception as e:
    st.error(f"Could not load the model. Check the model files and GPU setup. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a customer-support question...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

        if not answer:
            answer = "I couldn't generate a response. Please try rephrasing."

        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
