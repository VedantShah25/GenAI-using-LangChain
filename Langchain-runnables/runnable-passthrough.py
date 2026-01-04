from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

model=ChatOllama(model="mistral")
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Write a joke on {topic}.",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Explain the following joke:\n{text}",
    input_variables=["text"]
)

joke_chain=RunnableSequence(prompt1, model, parser)

parallel_chain=RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(prompt2, model, parser)
    }
)

final_chain=joke_chain | parallel_chain

result=final_chain.invoke({"topic":"cricket"})
print(result)