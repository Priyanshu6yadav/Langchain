import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# 1 Load local hugging face model

@st.cache_resource
def load_model():

    pipe = pipeline(
        'text-generation',
        model='HuggingFaceTB/SmolLM2-360M-Instruct',
        max_new_tokens=200, # max_new_tokens sets the strict upper limit on the number of tokens the model is allowed to generate in its response, excluding the prompt.
        temperature=0.7, # temperature controls the randomness of the output, 0.7 is a good default
        do_sample=True # do_sample=True means the model will sample from the distribution of possible tokens, 0.7 is a good default
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )
    return llm
llm = load_model()

# 2 Create LangChain Prompt

prompt = PromptTemplate.from_template(
    """
    You are an AI teacher.
    Explain the folling topic in simple language.
    Topic: {topic}
    
    Give:
    1. Definition
    2. Simple explanation
    3. Example"""
)

# 3. Create Chain
chain = prompt | llm

# 4. Streamlit Ui

st.title("🤖 Local LangChain Teacher")

topic = st.text_input(
    "Enter a topic",
    placeholder="Example: What is RAG?"
)


if st.button("Explain"):

    if topic:

        with st.spinner("Thinking..."):

            response = chain.invoke({
                "topic": topic
            })

        st.write(response)

    else:
        st.warning("Please enter a topic.")

