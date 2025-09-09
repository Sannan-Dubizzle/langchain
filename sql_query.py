from pydantic import BaseModel, Field


class SQLQuery(BaseModel):
    query: str = Field(
        description="The actual final SQL query"
    )