from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

model=ChatOllama(model="mistral")
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Write a joke on {topic}.",
    input_variables=["topic"]
)

joke_chain=RunnableSequence(prompt1, model, parser)

parallel_chain=RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(lambda x:len(x.split()))
    }
)

final_chain=joke_chain | parallel_chain

result=final_chain.invoke({"topic":"AI"})
print(result)

final_result="""{}\nWord Count: {}""".format(result["joke"], result["word_count"])
print(final_result)