from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch

model=ChatOllama(model="mistral")
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Summarize the following report in bullet points:\n{text}",    
    input_variables=["text"]
)

report_chain=RunnableSequence(prompt1, model, parser)

branch_chain=RunnableBranch(
    (lambda x: len(x) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

result=(report_chain | branch_chain).invoke({"topic":"Climate Change"})

print(result)