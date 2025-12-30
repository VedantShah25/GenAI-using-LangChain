from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import Literal

model = ChatOllama(model="mistral")

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="The overall sentiment of the feedback")
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Analyze the sentiment of the following feedback: {feedback}\n {format}",
    input_variables=["feedback"],
    partial_variables={"format": parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="You're receiving a feedback. Give an appropriate response to the following positive feedback: {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="You're receiving a feedback. Give an appropriate response to the following negative feedback: {feedback}",
    input_variables=["feedback"]
)

classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model | parser1),
    (lambda x: x.sentiment == "Negative", prompt3 | model | parser1),
    RunnableLambda(lambda x: "Sentiment not recognized.")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "The product quality is outstanding and exceeded my expectations!"})
print(result)