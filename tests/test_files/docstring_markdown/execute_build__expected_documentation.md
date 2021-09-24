- [/Users/shanekercheval/repos/python-helpers/tests/helpsk/example_class.py](#/Users/shanekercheval/repos/python-helpers/tests/helpsk/example_class.py)
    - [class ExampleClass](#class-ExampleClass)
        - [def my_method](#def-my_method)
        - [def my_variable](#def-my_variable)
        - [def my_variable](#def-my_variable)
- [/Users/shanekercheval/repos/python-helpers/tests/helpsk/utility.py](#/Users/shanekercheval/repos/python-helpers/tests/helpsk/utility.py)
    - [def is_debugging](#def-is_debugging)
- [/Users/shanekercheval/repos/python-helpers/tests/helpsk/string.py](#/Users/shanekercheval/repos/python-helpers/tests/helpsk/string.py)
    - [def collapse](#def-collapse)
    - [class RoundTo](#class-RoundTo)
    - [def format_number](#def-format_number)
- [/Users/shanekercheval/repos/python-helpers/tests/helpsk/date.py](#/Users/shanekercheval/repos/python-helpers/tests/helpsk/date.py)
    - [class Granularity](#class-Granularity)
    - [def floor](#def-floor)
    - [def fiscal_quarter](#def-fiscal_quarter)
    - [def to_string](#def-to_string)
- [/Users/shanekercheval/repos/python-helpers/tests/helpsk/validation.py](#/Users/shanekercheval/repos/python-helpers/tests/helpsk/validation.py)
    - [def any_none_nan](#def-any_none_nan)
    - [def assert_not_none_nan](#def-assert_not_none_nan)
    - [def any_missing](#def-any_missing)
    - [def assert_not_any_missing](#def-assert_not_any_missing)
    - [def any_duplicated](#def-any_duplicated)
    - [def assert_not_duplicated](#def-assert_not_duplicated)
    - [def assert_all](#def-assert_all)
    - [def assert_not_any](#def-assert_not_any)
    - [def assert_true](#def-assert_true)
    - [def assert_false](#def-assert_false)
    - [def raises_exception](#def-raises_exception)
    - [def dataframes_match](#def-dataframes_match)
        - [def first_dataframe_equals_other](#def-first_dataframe_equals_other)


## /Users/shanekercheval/repos/python-helpers/tests/helpsk/example_class.py

Documentation


### class ExampleClass

ExampleClass Documentation

init documentation

##### Args

    my_variable:
        docs


#### def my_method


```python
def my_method(self, value):
```

my_method Documentation


#### def my_variable


```python
def my_variable(self):
```

Getter method

#### def my_variable


```python
def my_variable(self, value):
```

Setter method

---

## /Users/shanekercheval/repos/python-helpers/tests/helpsk/utility.py

A collection of system and environment related helper functions.


### def is_debugging


```python
def is_debugging():
```

##### Returns

    Returns True if the environment is currently debugging (e.g. set a breakpoint in PyCharm),
    False if not


---

## /Users/shanekercheval/repos/python-helpers/tests/helpsk/string.py

Contains a collection of helper functions to manipulate strings.


### def collapse


```python
def collapse(*args: Union[str, List[str]], separate: str = '', surround: str = '') -> str:
```

Takes a list of strings and concatenates them, separating each string with the value of `separate` and
surrounding each string with the value of `surround`.

##### Args

    *args:
        list of strings to concatenate
    separate:
        string used to separate each string (default '')
    surround:
        string used to surround each string (default '')

##### Returns

    string surrounded by `surround` and separated by `separate`


### class RoundTo

Valid values for rounding numbers; used in e.g. `format_number()`


### def format_number


```python
def format_number(value: float,
              granularity: RoundTo = RoundTo.AUTO,
              places: int = 2) -> str:
```

Formats numbers in a way that humans would typically want to see at a glance (e.g. on a graph.)

For example, `123560000` is transformed to `123.56M`

##### Args

    value : the number (float) to format

    granularity : the granularity (e.g. thousands, millions, billions). (default is
        RoundTo.AUTO)

        `RoundTo.NONE` will not change the granularity, but will still add commas if necessary.

        `RoundTo.AUTO` will reduce the granularity (or not) depending on the number passed in.

    places: the number of digits to the right of the decimal to display. (default is `2`)

        For example:
            `format_number(123567000, granularity=RoundTo.MILLIONS, num_digits=2)` -> '123.57M'
            `format_number(123567000, granularity=RoundTo.MILLIONS, num_digits=3)` -> '123.567M'

##### Returns

    the formatted number as a string


---

## /Users/shanekercheval/repos/python-helpers/tests/helpsk/date.py

Contains a collection of date related helper functions.


### class Granularity

Valid values for date granularity


### def floor


```python
def floor(value: Union[datetime.datetime, datetime.date],
      granularity: Granularity = Granularity.DAY,
      fiscal_start: int = 1) -> datetime.date:

```

"Rounds" the datetime value down (i.e. floor) to the the nearest granularity.

For example, if `date` is `2021-03-20` and granularity is `Granularity.QUARTER` then `floor()` will return
`2021-01-01`.

If the fiscal year starts on November (`fiscal_start is `11`; i.e. the quarter is November,
December, January) and the date is `2022-01-28` and granularity is `Granularity.QUARTER` then `floor()`
will return `2021-11-01`.

##### Args

    value:
        a datetime

    granularity:
        the granularity to round down to (i.e. DAY, MONTH, QUARTER)

    fiscal_start:
        Only applicable for `Granularity.QUARTER`. The value should be the month index (e.g. Jan is 1,
        Feb, 2, etc.) that corresponds to the first month of the fiscal year.

##### Returns

    date - the date rounded down to the nearest granularity


### def fiscal_quarter


```python
def fiscal_quarter(value: Union[datetime.datetime, datetime.date],
               include_year: bool = False,
               fiscal_start: int = 1) -> float:
```

Returns the fiscal quarter (or year and quarter numeric value) for a given date.

For example:
    from dateutil.parser import parse

    date.fiscal_quarter(parse('2021-01-15')) == 1
    date.fiscal_quarter(parse('2021-01-15'), include_year=True) == 2021.1


    date.fiscal_quarter(parse('2021-01-15'), fiscal_start=2) == 4
    date.fiscal_quarter(parse('2021-01-15'), include_year=True, fiscal_start=2) == 2021.4
    date.fiscal_quarter(parse('2020-11-15'), include_year=True, fiscal_start=2) == 2021.4

Logic converted from R's Lubridate package:
    https://github.com/tidyverse/lubridate/blob/master/R/accessors-quarter.r

##### Args

    value:
        a date or datetime

    include_year:
        logical indicating whether or not to include the year
        if `True` then returns a float in the form of year.quarter.
        For example, Q4 of 2021 would be returned as `2021.4`

    fiscal_start:
        numeric indicating the starting month of a fiscal year.
        A value of 1 indicates standard quarters (i.e. starting in January).

##### Returns

    date - the date rounded down to the nearest granularity


### def to_string


```python
def to_string(value: Union[datetime.datetime, datetime.date],
          granularity: Granularity = Granularity.DAY,
          fiscal_start: int = 1) -> str:
```

Converts the date to a string.

Examples:

    ```
    from dateutil.parser import parse

    to_string(value=parse('2021-01-15'), granularity=Granularity.DAY) == "2021-01-15"
    to_string(value=parse('2021-01-15'), granularity=Granularity.Month) == "2021-Jan"
    to_string(value=parse('2021-01-15'), granularity=Granularity.QUARTER) == "2021-Q1"
    to_string(value=parse('2021-01-15'),
              granularity=Granularity.QUARTER,
              fiscal_start=2) == "2021-FQ4"
    ```

If the fiscal year starts on November (`fiscal_start is `11`; i.e. the quarter is November,
December, January) and the date is `2022-01-28` and granularity is `Granularity.QUARTER` then `floor()`
will return `2021-11-01`.

##### Args

    value:
        a datetime

    granularity:
        the granularity to round down to (i.e. DAY, MONTH, QUARTER)

    fiscal_start:
        Only applicable for `Granularity.QUARTER`. The value should be the month index (e.g. Jan is 1,
        Feb, 2, etc.) that corresponds to the first month of the fiscal year.

        If fiscal_start start is any other value than `1` quarters will be abbreviated with `F` to denote
        non-standard / fiscal quarters. For example, "2021-FQ4" is the 4th fiscal quarter of 2021.

##### Returns

    date - the date rounded down to the nearest granularity


---

## /Users/shanekercheval/repos/python-helpers/tests/helpsk/validation.py

A collection of functions that assist in validation/comparison of data and conditions.


### def any_none_nan


```python
def any_none_nan(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> bool:
```

Returns `True` if any item in `values` are `None`, `np.Nan`, or if the length of `values` is `0`.

For numeric types only.

##### Args

    values:
        A collection of values to check.

##### Returns

    bool - True if any item in `values` are None/np.NaN


### def assert_not_none_nan


```python
def assert_not_none_nan(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> None:
```

Raises an AssertionError if any item in `values` are `None`, `np.Nan`, or if the length of `values` is
`0`.

For numeric types only.

##### Args

    values:
        A collection of values to check.


### def any_missing


```python
def any_missing(values: Union[List, pd.Series, pd.DataFrame]) -> bool:
```

Returns `True` if any item in `values` are `None`, `np.Nan`, an empty string (i.e. '') or if the length
of `values` is `0`.

##### Args

    values:
        A collection of values to check.

##### Returns

    bool - True if any item in `values` are None/np.NaN/''


### def assert_not_any_missing


```python
def assert_not_any_missing(values: Union[List, pd.Series, pd.DataFrame]) -> None:
```

Raises an AssertionError if any item in `values` are `None`, `np.Nan`, an empty string (i.e. '') or if
the length of `values` is `0`.

##### Args

    values:
        A collection of values to check.


### def any_duplicated


```python
def any_duplicated(values: Union[List, np.ndarray, pd.Series]) -> bool:
```

Returns `True` if any items in `values` are duplicated.

##### Args

    values: list, np.ndarray, pd.Series
        A collection of values to check.

##### Returns

    bool


### def assert_not_duplicated


```python
def assert_not_duplicated(values: Union[List, np.ndarray, pd.Series]) -> None:
```

Raises an AssertionError if any items in `values` are duplicated.

##### Args

    values: list, np.ndarray, pd.Series
        A collection of values to check.


### def assert_all


```python
def assert_all(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> None:
```

Raises an `AssertionError` unless all items in `values` are `True`

##### Args

    values:
        A collection of values to check.


### def assert_not_any


```python
def assert_not_any(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> None:
```

Raises an `AssertionError` if any items in `values` are `True`

##### Args

    values:
        A collection of values to check.


### def assert_true


```python
def assert_true(condition: bool, message: str = 'Condition Not True') -> None:
```

Raises an AssertionError if `condition` is not True

##### Args

    condition: bool
        Something that evalualates to True/False


### def assert_false


```python
def assert_false(condition: bool, message: str = 'Condition True') -> None:
```

Raises an AssertionError if `condition` is not False

##### Args

    condition: bool
        Something that evalualates to True/False


### def raises_exception


```python
def raises_exception(function: Callable, exception_type: Type = None) -> bool:
```

Returns True if `function` raises an Exception; returns False if `function` runs without raising an
Exception.

##### Args

    function:
        the function which does or does not raise an exception.
    exception_type:
        if `exception_type` is provided, `raises_exception` returns true only if the `function` argument
        raises an Exception **and** the exception has type of `exception_type`.



### def dataframes_match


```python
def dataframes_match(dataframes: List[pd.DataFrame],
                 float_tolerance: int = 6,
                 ignore_indexes: bool = True,
                 ignore_column_names: bool = True) -> bool:
```

Because floating point numbers are difficult to accurate represent, when comparing multiple DataFrames,
this function first rounds any numeric columns to the number of decimal points indicated
`float_tolerance`.

##### Args

    dataframes:
        Two or more dataframes to compare against each other and test for equality

    float_tolerance:
        numeric columns will be rounded to the number of digits to the right of the decimal specified by
        this parameter.

    ignore_indexes:
        if True, the indexes of each DataFrame will be ignored for considering equality

    ignore_column_names:
        if True, the column names of each DataFrame will be ignored for considering equality

##### Returns

    Returns True if the dataframes match based on the conditions explained above, otherwise returns False


#### def first_dataframe_equals_other


```python
def first_dataframe_equals_other(other_dataframe):
    if first_dataframe.shape != other_dataframe.shape:
        return False

    if ignore_indexes or ignore_column_names:
        # if either of these are True, then we are going to change the index and/or columns, but
        # python is pass-by-reference so we don't want to change the original DataFrame object.
        other_dataframe = other_dataframe.copy()

    if ignore_indexes:
        other_dataframe.index = first_dataframe.index

    if ignore_column_names:
        other_dataframe.columns = first_dataframe.columns

    return first_dataframe.equals(other_dataframe.round(float_tolerance))

# compare the first dataframe to the rest of the dataframes, after rounding each to the tolerance, and
# performing other modifications
# check if all results are True
return all(first_dataframe_equals_other(x) for x in dataframes[1:])


def assert_dataframes_match(dataframes: List[pd.DataFrame],
                        float_tolerance: int = 6,
                        ignore_indexes: bool = True,
                        ignore_column_names: bool = True,
                        message: str = 'Dataframes do not match') -> None:
```

Raises an assertion error

##### Args

    dataframes:
        Two or more dataframes to compare against each other and test for equality

    float_tolerance:
        numeric columns will be rounded to the number of digits to the right of the decimal specified by
        this parameter.

    ignore_indexes:
        if True, the indexes of each DataFrame will be ignored for considering equality

    ignore_column_names:
        if True, the column names of each DataFrame will be ignored for considering equality

    message:
        message to pass to AssertionError


---

