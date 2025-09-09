from typing import List
from pydantic import BaseModel, Field
from typing import List


class Module(BaseModel):
    """A module representing a specific group of functionalities and tables"""

    name: str = Field(description="Name of module.")


def get_tables(category: Module) -> List[str]:
    tables = []
    if category.name == "orders":
        tables.extend(
            [
                "buyers",
                "users",
                "line_items",
                "variants",
                "products",
                "stores",
                "invoices",
                "payments",
                "digital_payments",
                "consignments",
                "orders",
                "courier_services",
                "locations",
                "statuses",
                "dispositions",
                "reviews",
                "comments"

            ]
        )
    elif category.name == "products":
        tables.extend(["products",
                       "variants",
                       "users",
                       "stores",
                       "line_items",
                       "reviews",
                       "comments"
                       "orders",
                       "buyers",
                       "categories",
                       "statuses",
                       "dispositions"
                       ])
    elif category.name == "stores&sellers/users":
        tables.extend(["stores",
                       "users",
                       "products",
                       "line_items",
                       "statuses",
                       "dispositions"
                       ])
    return tables
