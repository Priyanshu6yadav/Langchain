from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from transformers import pipeline

# 1. Setup local model pipeline
pipe = pipeline(
    task='text-generation',
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    max_new_tokens=250,
    return_full_text=False,
    device='mps'  # GPU acceleration on Mac Apple Silicon
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

# 2. Setup JSON Output Parser
parser = JsonOutputParser()

# 3. Prompt Template
# Tiny models need explicit instructions to only output the JSON object without python code or extra text
template = PromptTemplate(
    template="Return a valid JSON object with keys 'name', 'age', and 'city' for a fictional person. Do not write any explanations or code, only output the JSON object.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. Chain with LCEL
chain = template | model | parser

# Note: Even if input_variables is empty, invoke() requires an input dict: `{}`
result = chain.invoke({})

print("=== Parsed JSON Output (Python Dict) ===")
print(result)
print(f"Type: {type(result)}")
