from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Optional, Literal

class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["postive", "negative"] = Field(description="The overall sentiment of the review, either 'positive' or 'negative'")
    key_features: Optional[list[str]] = Field(default=None, description="A list of key features mentioned in the review")
    pros: Optional[list[str]] = Field(default=None, description="A list of pros mentioned in the review")
    cons: Optional[list[str]] = Field(default=None, description="A list of cons mentioned in the review")
    reviewer: Optional[str] = Field(default="Anonymous", description="Write the name of person who has written the review if not available then write 'Anonymous'")


model=ChatOllama(model="mistral")
structured_model=model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
""")

print(result)