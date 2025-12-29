from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

parser=JsonOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template="Give me name, age and networth (in millions) of a famous personality: {format}",
    input_variables=[],
    partial_variables={"format":parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({})
print(result)
print(type(result))