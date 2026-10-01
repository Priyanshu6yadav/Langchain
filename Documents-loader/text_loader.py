from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
prompt = PromptTemplate(
    template = "write a summary for the following text \n {poem}",
    input_variables = ['poem']
)
parser = StrOutputParser()

loader = TextLoader(
    "llm_history.txt",
    encoding = 'utf-8'
)

docs = loader.load()
# print(type(docs))
"""print(len(docs))
print(docs[0].metadata)
print(docs[0].page_content)"""

chain = prompt | model | parser
print(chain.invoke({'poem':docs[0].page_content}))