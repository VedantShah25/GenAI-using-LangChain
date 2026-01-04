from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

model = ChatOllama(model="mistral")

parser = StrOutputParser()
prompt1 = PromptTemplate(
    template="Write a joke on {topic}.",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Explain the following joke:\n{text}",
    input_variables=["text"]
)
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "nature"}))