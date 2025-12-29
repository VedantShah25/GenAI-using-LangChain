from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.structured import (StructuredOutputParser, ResponseSchema)
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact1", description="The first fact about the topic"),
    ResponseSchema(name="fact2", description="The second fact about the topic"),
    ResponseSchema(name="fact3", description="The third fact about the topic")
]

parser=StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="State 3 facts about {topic}. {format}",
    input_variables=["topic"],
    partial_variables={"format": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic": "Generative AI"})

print(result)