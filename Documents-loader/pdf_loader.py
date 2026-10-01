from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline


loader = PyPDFLoader(
    'Minor_Project_Synopsis_AI-SOC.pdf'
)
docs = loader.load()

# Hugging face model
pipe = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        device="mps",  # Use -1 for CPU
        max_new_tokens=200,
        do_sample=False,
)
llm = HuggingFacePipeline(pipeline = pipe)

prompt = PromptTemplate(
    template = "write the summary of this doctuments /n {docs}",
    input_variables=['docs']
)

parser = StrOutputParser()

chain = prompt | llm | parser

print(chain.invoke({'docs':docs}))