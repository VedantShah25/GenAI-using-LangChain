from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

model = ChatOllama(model="mistral")

parser = StrOutputParser()

prompt1=PromptTemplate(
    template="Write a tweet on {topic}.",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Write a linkedIn post on {topic}.",
    input_variables=["topic"]
)

chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkedIn_post": RunnableSequence(prompt2, model, parser)
    }   
)

result=chain.invoke({"topic":"Generative AI"})
print(result["tweet"])
print(result["linkedIn_post"])