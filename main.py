import getpass
import os
from pydantic import Field, BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# from langchain.chat_models import init_chat_model
#
# model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
#
# model.invoke("Hello, world!")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

article = '''
Let me be brutally honest with you.

There was a time I used to skip every video or blog that said “System Design”.
I’d think“This is for senior engineers, architects, not for me.”

I was wrong.

Because one day, in an interview, they asked me:
“Can you design a ride-sharing app like Uber?”
And I froze.

I talked about REST APIs.
I mentioned MySQL.
Then… silence.
No clue how to handle scale, no idea about queues, or even how to store real-time location.

That day I decided this won’t happen again.
Here’s how I went from being totally lost to confidently discussing architecture in interviews, and even proposing better designs at work.

1. First, I Accepted I Knew Nothing and That Was Okay
System Design is intimidating at first.
People throw words like “sharding”, “CQRS”, “load balancer”, “eventual consistency”…

At first, it made me feel dumb. But then I realized
Everyone feels lost in the beginning.

System design isn’t a single topic. It’s not a “chapter” you can complete in a week.
It’s a mix of:

How data flows
How services talk to each other
How systems survive under huge traffic
And how to make things fault-tolerant, fast, and reliable
Once I accepted that this will take time, it felt lighter.
I stopped chasing perfection and focused on small wins.

2. I Broke Down “System Design” Into Mini Topics
System Design is not one big subject it’s a set of interconnected building blocks.
So I made a map for myself:

a) The Basics
What happens when you type a URL in the browser
What is DNS, Load Balancer, CDN
TCP vs UDP, HTTP vs HTTPS
Even these basics were eye-opening. Like
Did you know DNS is like a phonebook of the internet? And CDNs are why YouTube loads fast?

b) Data and Storage
SQL vs NoSQL
Indexing, Replication, Sharding
When to choose MongoDB vs PostgreSQL
I learned this the hard way. In one project, we chose Mongo for transactional data. Later, we regretted it.

c) Scaling Techniques
Horizontal vs Vertical scaling
Caching (Redis, Memcached)
Load balancing (Round-robin, IP Hashing)
I loved this part. It made me feel like I could finally design something for millions of users even if it was just on paper.

d) Architecture Patterns
Monolith vs Microservices
Event-Driven Architecture
Pub/Sub, Message Queues (Kafka, RabbitMQ)
This made me understand why companies like Netflix use microservices not just because it’s trendy, but because it makes sense at scale.

3. I Watched Real People Think, Not Just Teach
Instead of watching tutorial-style videos, I started watching mock interviews.

And trust me, that changed everything.

Because when someone thinks aloud, makes mistakes, backtracks, and justifies their choices you learn how to think, not just copy.

Channels that really helped:

Gaurav Sen explains from the ground up
Exponent mock interviews with real candidates
ByteByteGo visual, storytelling approach
I learned how to:

Ask the right clarifying questions
Define functional and non-functional requirements
Walk through API design, DB choices, scaling logic
Always talk about tradeoffs, not just choices
4. I Started Drawing Even If It Was Just on Paper
'''

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a world class content writer that writes catchy titles for articles.")
user_prompt = HumanMessagePromptTemplate.from_template(
    '''You are given an atricle to analyze and create a catchy title, a seo friendly description
     and a concise summary for it.
    Following is the article to analyze:
    ___
    {article}
    ___
    The name should be based on the context of the article. Make it catchy and concise.
    Only output the title. Do not add any extra information.''',
    input_variables=["article"])

user_prompt.format(article=article)

first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])


class StrucutredResponse(BaseModel):
    title: str = Field(description=("The title of the article"))
    description: str = Field(description=("A short seo friendly description of the article"))
    summary: str = Field(description=("A concise summary of the article"))


structured_llm = llm.with_structured_output(StrucutredResponse)
chain_one = (
        {
            "article": lambda x: x['article']
        }
        | first_prompt
        | llm
        | {"article_title": lambda x: x.content}
)

chain_two = (
        {
            "article": lambda x: x['article']
        }
        | first_prompt
        | structured_llm
)


title = chain_one.invoke({"article": article})
print(title)

response = chain_two.invoke({"article": article})
print(response.title)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
connection = "postgresql+psycopg://sannan:@localhost:6024/langchain"

collection_name = "my_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]

vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

# delete from vector store
vector_store.delete(ids=["3"])

results = vector_store.similarity_search_with_score(
    "kitty", k=10
)
for doc in results:
    print(doc)
