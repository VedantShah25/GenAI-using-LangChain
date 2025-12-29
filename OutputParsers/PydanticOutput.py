from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str=Field(description="The name of the person")
    age: int=Field(ge=18, description="The age of the person")
    profession: str=Field(description="The profession of the person")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="Give name, age and profession of a famous {country} Celebrity. {format}",
    input_variables=[],
    partial_variables={"format":parser.get_format_instructions()}
)

# prompt = template.invoke({"country":"Indian"})

# print("Prompt:", prompt)

chain = template | model | parser
result = chain.invoke({"country":"Indian"})

print(result)
