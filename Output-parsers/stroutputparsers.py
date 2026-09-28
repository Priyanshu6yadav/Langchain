from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

# 1. Load local model using Hugging Face pipeline
# Using your downloaded model: SmolLM2-360M-Instruct (or "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
pipe = pipeline(
    task="text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    max_new_tokens=150,
    return_full_text=False,
    device="mps",  # GPU acceleration on Mac Apple Silicon
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

# 2. StrOutputParser automatically converts AIMessage -> clean str
parser = StrOutputParser()

# 3. Prompt Templates
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

#  --- Method 2: Step-by-Step with StrOutputParser ---
prompt1 = template1.invoke({'topic': 'black holes'})
result1 = model.invoke(prompt1)
report_text = parser.invoke(result1)  # StrOutputParser replaces `result1.content`
prompt2 = template2.invoke({'text': report_text})
result2 = model.invoke(prompt2)
summary_text = parser.invoke(result2)
print(summary_text)