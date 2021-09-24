# helpsk/example_class.py

Documentation
Documentation
Documentation
        """Getter method"""
        print("getter of my_variable called")
        return self._my_variable

    @my_variable.setter
    def my_variable(self, value):
# helpsk/utility.py

A collection of system and environment related helper functions.
Returns True if the environment is currently debugging (e.g. set a breakpoint in PyCharm), False if not

Returns
-------
bool
# helpsk/string.py

Contains a collection of helper functions to manipulate strings.
Takes a list of strings and concatenates them, separating each string with the value of `separate` and
surrounding each string with the value of `surround`.

Keyword arguments:
*args -- list of strings to concatenate
separate -- string used to separate each string (default '')
surround -- string used to surround each string (default '')
Valid values for rounding numbers; used in e.g. `format_number()`
Formats numbers in a way that humans would typically want to see at a glance (e.g. on a graph.)

For example, `123560000` is transformed to `123.56M'

Parameters
----------
value : the number (float) to format

granularity : the granularity (e.g. thousands, millions, billions). (default is
    RoundTo.AUTO)

    `RoundTo.NONE` will not change the granularity, but will still add commas if necessary.

    `RoundTo.AUTO` will reduce the granularity (or not) depending on the number passed in.

places: the number of digits to the right of the decimal to display. (default is `2`)

    For example:
        `format_number(123567000, granularity=RoundTo.MILLIONS, num_digits=2)` -> '123.57M'
        `format_number(123567000, granularity=RoundTo.MILLIONS, num_digits=3)` -> '123.567M'

Returns
-------
the formatted number as a string
# helpsk/date.py

Contains a collection of date related helper functions.
Valid values for date granularity
Takes a string in the form of 'YYYY-MM-DD' and returns a datetime.date

Parameters
----------
yyyy_mm_dd : str
    a string in the form of 'YYYY-MM-DD'

Returns
-------
datetime.date
Takes a string in the form of 'YYYY-MM-DD HH:MM:SS' and returns a datetime.datetime

Parameters
----------
yyyy_mm_dd_hh_mm_ss : str
    a string in the form of 'YYYY-MM-DD HH:MM:SS'

Returns
-------
datetime.datetime
"Rounds" the datetime value down (i.e. floor) to the the nearest granularity.

For example, if `date` is `2021-03-20` and granularity is `Granularity.QUARTER` then `floor()` will return
`2021-01-01`.

If the fiscal year starts on November (`fiscal_start is `11`; i.e. the quarter is November,
December, January) and the date is `2022-01-28` and granularity is `Granularity.QUARTER` then `floor()`
will return `2021-11-01`.

Parameters
----------
value : datetime.datetime
    a datetime

granularity: Granularity
    the granularity to round down to (i.e. DAY, MONTH, QUARTER)

fiscal_start: int
    Only applicable for `Granularity.QUARTER`. The value should be the month index (e.g. Jan is 1, Feb, 2,
    etc.) that corresponds to the first month of the fiscal year.

Returns
-------
date - the date rounded down to the nearest granularity
Returns the fiscal quarter (or year and quarter numeric value) for a given date.

For example:
    date.fiscal_quarter(date.ymd('2021-01-15')) == 1
    date.fiscal_quarter(date.ymd('2021-01-15'), include_year=True) == 2021.1


    date.fiscal_quarter(date.ymd('2021-01-15'), fiscal_start=2) == 4
    date.fiscal_quarter(date.ymd('2021-01-15'), include_year=True, fiscal_start=2) == 2021.4
    date.fiscal_quarter(date.ymd('2020-11-15'), include_year=True, fiscal_start=2) == 2021.4

Logic converted from R's Lubridate package:
    https://github.com/tidyverse/lubridate/blob/master/R/accessors-quarter.r

Parameters
----------
value : datetime.datetime
    a date or datetime

include_year: bool
    logical indicating whether or not to include the year
    if `True` then returns a float in the form of year.quarter.
    For example, Q4 of 2021 would be returned as `2021.4`

fiscal_start: int
    numeric indicating the starting month of a fiscal year.
    A value of 1 indicates standard quarters (i.e. starting in January).

Returns
-------
date - the date rounded down to the nearest granularity
Converts the date to a string.

Examples:
    to_string(value=ymd('2021-01-15'), granularity=Granularity.DAY) == "2021-01-15"
    to_string(value=ymd('2021-01-15'), granularity=Granularity.Month) == "2021-Jan"
    to_string(value=ymd('2021-01-15'), granularity=Granularity.QUARTER) == "2021-Q1"
    to_string(value=ymd('2021-01-15'),
              granularity=Granularity.QUARTER,
              fiscal_start=2) == "2021-FQ4"

If the fiscal year starts on November (`fiscal_start is `11`; i.e. the quarter is November,
December, January) and the date is `2022-01-28` and granularity is `Granularity.QUARTER` then `floor()`
will return `2021-11-01`.

Parameters
----------
value : datetime.datetime
    a datetime

granularity: Granularity
    the granularity to round down to (i.e. DAY, MONTH, QUARTER)

fiscal_start: int
    Only applicable for `Granularity.QUARTER`. The value should be the month index (e.g. Jan is 1, Feb, 2,
    etc.) that corresponds to the first month of the fiscal year.

    If fiscal_start start is any other value than `1` quarters will be abbreviated with `F` to denote
    non-standard / fiscal quarters. For example, "2021-FQ4" is the 4th fiscal quarter of 2021.

Returns
-------
date - the date rounded down to the nearest granularity
# helpsk/validation.py

A collection of functions that assist in validation/comparison of data and conditions.
Returns `True` if any item in `values` are `None`, `np.Nan`, or if the length of `values` is `0`.
For numeric types only.

Parameters
----------
values : list, np.ndarray, pd.Series, pd.DataFrame
    A collection of values to check.

Returns
-------
bool - True if any item in `values` are None/np.NaN
Raises an AssertionError if any item in `values` are `None`, `np.Nan`, or if the length of `values` is
`0`.
For numeric types only.

Parameters
----------
values : list, np.ndarray, pd.Series, pd.DataFrame
    A collection of values to check.

Returns
-------
None
Returns `True` if any item in `values` are `None`, `np.Nan`, an empty string (i.e. '') or if the length of
`values` is `0`.

Parameters
----------
values : list, pd.Series, pd.DataFrame
    A collection of values to check.

Returns
-------
bool - True if any item in `values` are None/np.NaN/''
Raises an AssertionError if any item in `values` are `None`, `np.Nan`, an empty string (i.e. '') or if the
length of `values` is `0`.

Parameters
----------
values : list, pd.Series, pd.DataFrame
    A collection of values to check.

Returns
-------
bool - True if any item in `values` are None/np.NaN/''
Returns `True` if any items in `values` are duplicated.

Parameters
----------
values : list, np.ndarray, pd.Series
    A collection of values to check.

Returns
-------
bool
Raises an AssertionError if any items in `values` are duplicated.

Parameters
----------
values : list, np.ndarray, pd.Series
    A collection of values to check.

Returns
-------
Raises an `AssertionError` unless all items in `values` are `True`

Parameters
----------
values : list, np.ndarray, pd.Series, pd.DataFrame
    A collection of values to check.

Returns
-------
None
Raises an `AssertionError` if any items in `values` are `True`

Parameters
----------
values : list, np.ndarray, pd.Series, pd.DataFrame
    A collection of values to check.

Returns
-------
None
Raises an AssertionError if `condition` is not True
Raises an AssertionError if `condition` is not False
Returns True if `function` raises an Exception; returns False if `function` runs without raising an
Exception.

Keyword arguments:
function -- a function
Because floating point numbers are difficult to accurate represent, when comparing multiple DataFrames,
this function first rounds any numeric columns to the number of decimal points indicated
`float_tolerance`.

Parameters
----------
dataframes : list of pd.DataFrame
    Two or more dataframes to compare against each other and test for equality

float_tolerance: int
    numeric columns will be rounded to the number of digits to the right of the decimal specified by
    this parameter.

ignore_indexes: bool
    if True, the indexes of each DataFrame will be ignored for considering equality

ignore_column_names: bool
    if True, the column names of each DataFrame will be ignored for considering equality
Raises an assertion error

Parameters
----------
dataframes : list of pd.DataFrame
    Two or more dataframes to compare against each other and test for equality

float_tolerance: int
    numeric columns will be rounded to the number of digits to the right of the decimal specified by
    this parameter.

ignore_indexes: bool
    if True, the indexes of each DataFrame will be ignored for considering equality

ignore_column_names: bool
    if True, the column names of each DataFrame will be ignored for considering equality

message: str
    message to pass to AssertionError
