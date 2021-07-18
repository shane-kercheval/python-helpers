from typing import List, Union, Callable


def raises_exception(function: Callable):
    """Returns True if `function` raises an Exception; returns False if `function` runs without raising an Exzception.

    Keyword arguments:
    function -- a function
    """
    try:
        function()
        return False
    except:  # noqa
        return True


def any_duplicated(thing):
    raise NotImplementedError()


def assert_not_any(thing):
    """Raises Exception if any values are true
    """
    raise NotImplementedError()


def assert_identical(thing):
    """Raises Exception if xyz is not identical
    """    
    raise NotImplementedError()


def assert_none_missing(thing, empty_string_as_missing: bool = True):
    """Raises Exception if any values are missing.

    Keyword arguments:
    empty_string_as_missing -- if True, treats empty string as missing value
    """
    raise NotImplementedError()


def assert_none_duplicated(thing, remove_missing_values: bool = True):
    """Raises Exception if any values are missing.

    Keyword arguments:
    remove_missing_values -- if True, removes missing values before checking if duplicated
    """

    # if remove_missing_values is False throw exception if more than one are missing?
    raise NotImplementedError()


 @staticmethod
def assert_dataframes_equal(data_frame1: pd.DataFrame,
                            data_frame2: pd.DataFrame,
                            check_column_types: bool = True):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # check that the types of the columns are all the same
    if check_column_types:
        assert all([x == y for x, y in zip(data_frame1.dtypes.values, data_frame2.dtypes.values)])
    assert all(data_frame1.columns.values == data_frame2.columns.values)
    assert all(data_frame1.index.values == data_frame2.index.values)
    numeric_col, cat_cols = OOLearningHelpers.get_columns_by_type(data_dtypes=data_frame1.dtypes)

    for col in numeric_col:
        # check if the values are close, or if they are both NaN
        assert all([isclose(x, y) or (math.isnan(x) and math.isnan(y))
                    for x, y in zip(data_frame1[col].values, data_frame2[col].values)])

    for col in cat_cols:
        # if the two strings aren't equal, but also aren't 'nan', it will cause a problem because
        # isnan will try to convert the string to a number, but it will fail with TypeError, so have to
        # ensure both values are a number before we check that they are nan.
        assert all([x == y or (is_number(x) and is_number(y) and math.isnan(x) and math.isnan(y))
                    for x, y in zip(data_frame1[col].values, data_frame2[col].values)])

    return True
