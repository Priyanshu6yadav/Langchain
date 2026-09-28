from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

# 1. Load local model using Hugging Face pipeline
# Using your downloaded model: SmolLM2-360M-Instruct (or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
pipe = pipeline(
    task="text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    max_new_tokens=250,
    return_full_text=False,
    device="mps",  # GPU acceleration on Mac Apple Silicon
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

# 1st prompt --> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt --> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=['text']
)
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'black holes'})
print(result) 