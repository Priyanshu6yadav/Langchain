from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 1. Define Pydantic Schema
class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")

# 2. Setup Pydantic Output Parser
parser = PydanticOutputParser(pydantic_object=Person)

# 3. Setup Model
# Note: PydanticOutputParser injects a complex JSON schema in get_format_instructions().
# Cloud models like Groq follow this schema reliably.
model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# 4. Prompt Template
template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person.\n{format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# 5. Chain with LCEL
chain = template | model | parser
final_result = chain.invoke({'place': 'India'})

print("=== Parsed Pydantic Object ===")
print(final_result)
print(f"Type: {type(final_result)}")
print(f"name: {final_result.name}, age: {final_result.age}, city: {final_result.city}")