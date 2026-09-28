
import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

st.title("Local Hugging Face AI")
st.caption("Runs locally — no paid inference API")

@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        device="mps",  # Use -1 for CPU
        max_new_tokens=200,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

prompt = PromptTemplate.from_template(
    "You are a helpful AI assistant.\n"
    "Question: {question}\n"
    "Answer:"
)

chain = prompt | llm

question = st.text_input("Ask a question")

if st.button("Generate") and question:
    with st.spinner("Generating locally..."):
        answer = chain.invoke({"question": question})
    st.write(answer)
