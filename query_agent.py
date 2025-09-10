import os
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from my_tools import get_modules, get_relevant_tables_schema, QuerySQLCheckerTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
)

from langgraph.checkpoint.memory import MemorySaver

from prompts import POSTGRES_PROMPT



def get_db() -> SQLDatabase:
    return SQLDatabase.from_uri(
        os.environ.get("DATABASE_URI"))


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )


def get_memory_saver() -> MemorySaver:
    return MemorySaver()


def get_available_tools(db: SQLDatabase, llm: ChatOpenAI) -> []:
    db_tool = QuerySQLDatabaseTool(db=db)
    query_checker_tool = QuerySQLCheckerTool(db=db, llm=llm)
    return [
        get_modules,
        get_relevant_tables_schema,
        db_tool,
        query_checker_tool
    ]


def get_agent_executor() -> CompiledStateGraph:
    db = get_db()
    llm = get_llm()

    system_story = """This is an online e-commerce platform. It allows a user to create a store and list products for selling.
    Buyers can purchase the products. Third Party Logistics companies alias courier services deliver these products to the buyers.
    Buyers pay the amount which is equal to orders' net amount. From net amount, the share of courier services is deducted and paid to them.
    Fee is deducted and paid to the platform which is the actual profit of the platform. And the rest is paid to the seller(user) which is the income of the seller.
       """
    prompt_template = ChatPromptTemplate.from_template(system_story + POSTGRES_PROMPT)

    memory = get_memory_saver()
    return create_react_agent(llm, get_available_tools(db, llm),
                              checkpointer=memory,
                              prompt=prompt_template.format(top_k=6))
