
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# Hugging Face model
model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Load model locally
pipe = pipeline(
    task="text-generation",
    model=model_id,
    max_new_tokens=200,
    do_sample=False,
    device="mps",  # Apple Silicon Mac
)

# Connect model to LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Create a prompt template
prompt = PromptTemplate.from_template(
    "You are a helpful AI assistant.\n"
    "Question: {question}\n"
    "Answer:"
)

# Create a chain
chain = prompt | llm

# Ask a question
response = chain.invoke({
    "question": "Explain RAG in simple words."
})

print(response)
