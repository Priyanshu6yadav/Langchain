# A structured output parser is a tool that forces an AI to reply in a neat, predictable format (like JSON) instead of plain text, making it easy for your code to read.

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
# In LangChain 1.x, StructuredOutputParser and ResponseSchema live in langchain_classic.output_parsers
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

# 1. Setup local model pipeline
pipe = pipeline(
    task='text-generation',
    model='HuggingFaceTB/SmolLM2-360M-Instruct',  # Fast, highly accurate instruction-following model
    max_new_tokens=250,
    return_full_text=False,
    do_sample=False,  # Greedy decoding ensures deterministic, valid JSON output
    device='mps'      # GPU acceleration on Mac Apple Silicon
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

# 2. Define the schema
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# Note: The method name is `from_response_schemas` (plural)
parser = StructuredOutputParser.from_response_schemas(schema)

# 3. Prompt Template
template = PromptTemplate(
    template="Give 3 facts about {topic}.\n{format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'apple'})
print(result)