"""This module contains classes that wrap the connection/querying logic of various databases.
Database objects can be created from the __init__ function or from the .from_config class function
by passing in an instance of the corresponding configuration class.

Examples:

    with Redshift(user='username', password='asdf', ....) as redshift:
        redshift.query("SELECT * FROM table LIMIT 100")

    or

    redshift_config = RedshiftConfigFile('/path/to/redshift.config')
    with Redshift.from_config(redshift_config) as redshift:
        redshift.query("SELECT * FROM table LIMIT 100")
"""
from __future__ import annotations
import pandas as pd
from helpsk.database_base import Database, ConnectionObject


class Redshift(Database):
    """
    Wraps logic for connecting to Redshift and querying.

    Example:
        Given redshift.config with contents:

            [redshift]
            dbname=my_username
            password=my-password-123
            database=the_database
            port=1234
            host=host.address.redshift.amazonaws.com

        with Redshift.from_config('/path/to/redshift.config', 'redshift') as redshift:
            redshift.query("SELECT * FROM table LIMIT 100")

        or

        with Redshift(dbname='my_username', password=...) as redshift:
            redshift.query("SELECT * FROM table LIMIT 100")
    """

    @property
    def _connection_string(self) -> str:
        return ' '.join(f"{k}={v}" for k, v in self._kwargs.items())

    def _open_connection_object(self) -> ConnectionObject:
        """Wraps logic for connecting to redshift"""
        from psycopg2 import connect
        return connect(self._connection_string)

    def _close_connection_object(self):
        """Wraps logic for closing the connection to redshift"""
        self.connection_object.close()

    def _query(self, sql: str) -> pd.DataFrame:
        """Wraps logic for querying redshift.

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"

        Returns:
            a pandas Dataframe with the results from the query
        """
        return pd.read_sql_query(sql, self.connection_object)

    def execute_statement(self, statement: str):
        raise NotImplementedError()

    def insert_records(self,
                       dataframe: pd.DataFrame, table: str, create_table: bool = False,
                       overwrite: bool = True, schema: str = None,
                       database: str = None):
        raise NotImplementedError()


class Snowflake(Database):
    """Wraps logic for connecting to Snowflake and querying.

        Example:
            config = SnowflakeConfigFile('/path/to/snowflake.config')
            with Snowflake.from_config(config) as snowflake:
                snowflake.query("SELECT * FROM table LIMIT 100")

    Instructions for installing snowflake dependencies:
        https://docs.snowflake.com/en/user-guide/python-connector-install.html

    I used v2.7.4, and python v3.9 (requirements_39):

    This page shows the latest versions:
        https://pypi.org/project/snowflake-connector-python/

    ```
    pip install -r .../snowflake-connector-python/v2.7.4/tested_requirements/requirements_39.reqs
    pip install snowflake-connector-python==v2.7.4
    ```

    Additionally:
        https://docs.snowflake.com/en/user-guide/python-connector-pandas.html#installation

    ```
    pip install snowflake-connector-python[pandas]
    ```
    """
    def __init__(self, **kwargs):
        """Initialization"""
        super().__init__(**kwargs)

        if (autocommit := kwargs.get('autocommit')) is not None and isinstance(autocommit, str):
            if autocommit.lower() == 'false':
                self._kwargs['autocommit'] = False
            elif autocommit.lower() == 'true':
                self._kwargs['autocommit'] = True
            else:
                raise ValueError(f"autocommit needs to be true/false but received `{autocommit}`")

    def _open_connection_object(self) -> ConnectionObject:
        """Wraps logic for connecting to snowflake"""
        from snowflake.connector import connect
        return connect(**self._kwargs)

    def _close_connection_object(self):
        """Wraps logic for closing the connection to snowflake
        """
        self.connection_object.close()

    def _query(self, sql: str) -> pd.DataFrame:
        """Wraps logic for querying snowflake.

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"

        Returns:
            a pandas Dataframe with the results from the query
        """
        cursor = self.connection_object.cursor()
        cursor.execute(sql)
        dataframe = cursor.fetch_pandas_all()
        # We need to reset_index because there seems to be a bug in the connector that returns
        # duplicated index values. This can cause unexpected behavior downstream.
        # https://github.com/snowflakedb/snowflake-connector-python/issues/1061
        # https://stackoverflow.com/questions/69911999/none-unique-pandas-dataframe-index-created-using-cur-fetch-pandas-all-after-lo
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    def execute_statement(self, statement: str):
        """This method executes a statement without any data returned."""
        cursor = self.connection_object.cursor()
        results = cursor.execute(statement)
        results = results.fetchall()
        return results

    @staticmethod
    def _generate_sql_create_table(table: str, dataframe: pd.DataFrame,
                                   database: str = None, schema: str = None):
        # adapted from https://stephenallwright.com/create-snowflake-table-pandas-dataframe/
        final_table_name = ''
        if database:
            final_table_name += f'{database.upper()}.'
        if schema:
            final_table_name += f'{schema.upper()}.'
        final_table_name += table.upper()

        create_statement = f"CREATE OR REPLACE TABLE {final_table_name} (\n"
        # Loop through each column finding the datatype and adding it to the statement
        for column in dataframe.columns:
            if dataframe[column].dtype.name == 'int' or dataframe[column].dtype.name == 'int64':
                create_statement += f"    {column} int"
            elif dataframe[column].dtype.name == 'object':
                create_statement += f"    {column} varchar"
            elif dataframe[column].dtype.name == 'datetime64[ns]':
                create_statement += f"    {column} datetime"
            elif dataframe[column].dtype.name == 'float64':
                create_statement += f"    {column} float8"
            elif dataframe[column].dtype.name == 'bool':
                create_statement += f"    {column} boolean"
            else:
                create_statement += f"    {column} varchar"

            # If column is not last column, add comma, else end sql-query
            if dataframe[column].name != dataframe.columns[-1]:
                create_statement += ",\n"

        create_statement += "\n)"
        return create_statement

    def insert_records(self,
                       dataframe: pd.DataFrame,
                       table: str,
                       create_table: bool = False,
                       overwrite: bool = False,
                       schema: str = None,
                       database: str = None):
        """
        This method inserts rows into a table from a pandas DataFrame.

        Args:
            dataframe:
                the pandas dataframe to insert
            table:
                the name of the table
            create_table:
                if True, creates the table before inserting
            overwrite:
                if True, drops all records before inserting
            schema:
                the name of the schema
            database:
                the name of the database
        """
        from snowflake.connector.pandas_tools import write_pandas
        dataframe = dataframe.copy()
        # snowflake is case sensitive and converts everything to upper-case
        dataframe.columns = dataframe.columns.str.upper()

        if create_table:
            create_table_sql = Snowflake._generate_sql_create_table(
                table=table,
                dataframe=dataframe,
                database=database,
                schema=schema,
            )
            _ = self.execute_statement(statement=create_table_sql)

        # if we are creating the table there is nothing to overwrite
        if not create_table and overwrite:
            final_table_name = ''
            if database:
                final_table_name += f'{database.upper()}.'
            if schema:
                final_table_name += f'{schema.upper()}.'
            final_table_name += table.upper()

            _ = self.execute_statement(f"DELETE FROM {final_table_name}")

        if database:
            database = database.upper()

        if schema:
            schema = schema.upper()

        _ = self.execute_statement(statement=f"USE SCHEMA {schema};")
        success, _, num_rows, _ = write_pandas(
            conn=self.connection_object,
            df=dataframe,
            database=database,
            schema=schema,
            table_name=table.upper()
        )
        return success, num_rows


class Sqlite(Database):
    """
    Wraps logic for connecting to Redshift and querying.

        with Sqlite(path='/path/to/sqlite.db') as db:
            db.query("SELECT * FROM table LIMIT 100")
    """

    @property
    def _connection_string(self) -> str:
        return f"sqlite:///{self._kwargs['path']}"

    def _open_connection_object(self) -> ConnectionObject:
        """Wraps logic for connecting to sqlite"""
        from sqlalchemy import create_engine
        engine = create_engine(self._connection_string)
        return engine.connect()

    def _close_connection_object(self):
        """Wraps logic for closing the connection to sqlite"""
        self.connection_object.close()

    def _query(self, sql: str) -> pd.DataFrame:
        """Wraps logic for querying redshift.

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"

        Returns:
            a pandas Dataframe with the results from the query
        """
        return pd.read_sql(sql, self.connection_object)

    def execute_statement(self, statement: str):
        """This method executes a statement without any data returned."""
        self.connection_object.execute(statement)

    def insert_records(self,
                       dataframe: pd.DataFrame,
                       table: str,
                       create_table: bool = False,
                       overwrite: bool = False,
                       schema: str = None,
                       database: str = None):
        """
        This method inserts rows into a table from a pandas DataFrame.

        Args:
            dataframe:
                the pandas dataframe to insert
            table:
                the name of the table
            create_table:
                if True, creates the table before inserting
            overwrite:
                if True, drops all records before inserting
            schema:
                the name of the schema
            database:
                Not used.
        """
        # if_exists{‘fail’, ‘replace’, ‘append’}, default ‘fail’
        #     How to behave if the table already exists.
        #     fail: Raise a ValueError.
        #     replace: Drop the table before inserting new values.
        #     append: Insert new values to the existing table.
        if create_table:
            if_exists = 'fail'
        elif overwrite:
            if_exists = 'replace'
        else:
            if_exists = 'append'

        _ = dataframe.to_sql(
            name=table,
            con=self.connection_object,
            schema=schema,
            if_exists=if_exists,
            index=False
        )
