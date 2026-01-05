from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(model="mistral")
parser = StrOutputParser()

loader=TextLoader("Essay_on_Cricket.txt", encoding="utf-8")
docs=loader.load()

# print(type(docs))
# print(len(docs))
# print(type(docs[0].page_content))
# print(docs[0].metadata)

prompt=PromptTemplate(
    template="Suggest what improvements can be made in the given essay - {essay}.",
    input_variables=["essay"]
)

chain = prompt | model | parser

result= chain.invoke({"essay": docs[0].page_content})
print(result)