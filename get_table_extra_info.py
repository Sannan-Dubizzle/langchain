def get_tables_extra_info(table_names: list[str]) -> str:
    """Accepts the list of table names in the snake_underscores format and returns the extra information of a database table"""
    extra_info = ""
    for table_name in table_names:
        if get_table_extra_info(table_name) is not None:
            extra_info += "{table_name} - {info} \n".format(table_name=table_name,
                                                            info=get_table_extra_info(table_name))
    return extra_info


def get_table_extra_info(table_name: str):
    return table_extra_info.get(table_name)


table_extra_info = {
    # this will contain any extra info to provide about the tables for better context
   }
