from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import Literal

load_dotenv()
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text as positive or negative.\n\n{format_instructions}\n\nFeedback:\n{feedback}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

# FIXED: Changed 'temperature' to 'template'
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

# FIXED: Corrected RunnableBranch spelling, parser variable name, and input mapping
branch_chain = RunnableBranch(
    (lambda res: res['sentiment'].sentiment == 'positive', RunnableLambda(lambda x: x['input']) | prompt2 | model | parser),
    (lambda res: res['sentiment'].sentiment == 'negative', RunnableLambda(lambda x: x['input']) | prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

# Combine using a dictionary pass-through so both original input and sentiment are available
full_chain = (
    RunnableLambda(lambda x: {'sentiment': classifier_chain.invoke(x), 'input': x})
    | branch_chain
)

print(full_chain.invoke({'feedback': 'this is a terrible phone'}))