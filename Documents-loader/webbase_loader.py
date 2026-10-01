from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Initialize Groq model
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
prompt = PromptTemplate(
    template='Answer the following question \n {question} from the dollowing text - \n {text}',
    input_variables = ['question','text']
)
parser = StrOutputParser()


url = "https://www.flipkart.com/apple-iphone-17-pro-max-cosmic-orange-1-tb/p/itm1f80bd9e45636?pid=MOBHFN6YEZHFFZ6Q&lid=LSTMOBHFN6YEZHFFZ6QTDLV62&hl_lid=&marketplace=FLIPKART&fm=eyJ3dHAiOiJyZWNvIiwicHJwdCI6InBwIiwibWlkIjoicHJvZHVjdFJlY29tbWVuZGF0aW9uL2FzcGVjdFNpbWlsYXIifQ%3D%3D&pageUID=1790834085031"
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'explain the product processor performance','text':docs[0].page_content}))
