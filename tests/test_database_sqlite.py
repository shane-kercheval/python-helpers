import os
import pytest
from sqlalchemy.exc import OperationalError

from helpsk.database import Sqlite
from helpsk.validation import dataframes_match

import os.path


def test_sqlite():
    import pandas as pd
    df = pd.DataFrame({
        'column_1': ['a', None, 'c'],
        'column_2': [1, 2, None],
    })

    db_path = 'tests/test.db'
    try:
        assert not os.path.exists(db_path)
        with Sqlite(path=db_path) as db:
            assert os.path.exists(db_path)

            with pytest.raises(OperationalError):
                db.query("SELECT * FROM TEST_TABLE")

            db.insert_records(
                dataframe=df,
                table='TEST_TABLE',
                create_table=False,
                overwrite=False,
            )
            test_table_df = db.query("SELECT * FROM TEST_TABLE")
            assert dataframes_match([df, test_table_df])

            db.insert_records(
                dataframe=df,
                table='TEST_TABLE',
                create_table=False,
                overwrite=True,
            )
            test_table_df = db.query("SELECT * FROM TEST_TABLE")
            assert dataframes_match([df, test_table_df])

            with pytest.raises(ValueError):
                db.insert_records(
                    dataframe=df,
                    table='TEST_TABLE',
                    create_table=True,
                    overwrite=False,
                )
            expected_df = pd.concat([df, df]).reset_index(drop=True)
            db.insert_records(dataframe=df, table='TEST_TABLE')
            test_table_df = db.query("SELECT * FROM TEST_TABLE")
            assert dataframes_match([expected_df, test_table_df])

        with Sqlite(path=db_path) as db:
            test_table_df = db.query("SELECT * FROM TEST_TABLE")
            assert dataframes_match([expected_df, test_table_df])

            db.execute_statement("DELETE FROM TEST_TABLE WHERE column_1 = 'a'")
            test_table_df = db.query("SELECT * FROM TEST_TABLE")
            assert dataframes_match([
                test_table_df,
                expected_df[expected_df['column_1'] != 'a'].reset_index(drop=True)
            ])

            db.execute_statement("DROP TABLE TEST_TABLE;")
            with pytest.raises(OperationalError):
                db.query("SELECT * FROM TEST_TABLE")
    finally:
        os.remove(db_path)
