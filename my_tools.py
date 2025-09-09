from langchain_core.tools import tool
from typing import List, Optional
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field, ConfigDict, model_validator
from langchain_core.tools import BaseTool
from get_table_extra_info import get_table_extra_info
from langchain_community.utilities.sql_database import SQLDatabase
from prompts import QUERY_CHECKER
from sql_databse_mapper import db_schema
from tables import Module, get_tables
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


@tool
def get_tables_info(table_names: Optional[List[str]] = []) -> str:
    """This tool accepts the list of table names in form of list of stings and returns the db_schema and project level
    information about those tables in form of a single string"""
    schema = ""
    for table_name in table_names:
        schema += db_schema.get(table_name) if db_schema.get(table_name) else "" + "\n"
        schema += str(
            get_table_extra_info(table_name)) + "\n\n\n\n" if get_table_extra_info(table_name) else ""

    return schema


@tool
def get_relevant_tables(module: Module) -> List[str]:
    """accepts the module and returns the list of database table names relating to that module"""
    return get_tables(module)


@tool
def get_modules() -> List[Module]:
    """ returns the list of available modules of the application"""
    var = [
        Module(name="orders"),
        Module(name="products"),
        Module(name="stores&sellers/users")
    ]
    return var


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description="A detailed and SQL query to be checked.")


class QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct.
        Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = "sql_db_query_checker"
    description: str = """
            Use this tool to double check if your query is correct before executing it.
            Always use this tool before executing a query with sql_db_query!
            """
    args_schema: Type[BaseModel] = _QuerySQLCheckerToolInput

    @model_validator(mode="before")
    @classmethod
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Any:
        if "llm_chain" not in values:
            from langchain.chains.llm import LLMChain

            values["llm_chain"] = LLMChain(llm=values.get("llm"),
                                           prompt=PromptTemplate(template=QUERY_CHECKER,
                                                                 input_variables=["dialect", "query"]))

        if values["llm_chain"].prompt.input_variables != ["dialect", "query"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
            )

        return values

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(
            query=query,
            dialect=self.db.dialect,
            callbacks=run_manager.get_child() if run_manager else None,
        )

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.apredict(
            query=query,
            dialect=self.db.dialect,
            callbacks=run_manager.get_child() if run_manager else None,
        )
