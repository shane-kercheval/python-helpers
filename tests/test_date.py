import unittest

import pandas as pd

from helpsk import date
from helpsk import validation as vld


# noinspection PyMethodMayBeStatic
class TestValidation(unittest.TestCase):

    def test_ymd(self):
        value = date.ymd('2021-01-01')
        assert value.year == 2021
        assert value.month == 1
        assert value.day == 1

        value = date.ymd('2021-01-31')
        assert value.year == 2021
        assert value.month == 1
        assert value.day == 31

        value = date.ymd('2021-12-31')
        assert value.year == 2021
        assert value.month == 12
        assert value.day == 31

        assert vld.raises_exception(lambda: date.ymd('2021-02-31'), ValueError)

    def test_date_to_string(self):
        pass

    def test_floor_month(self):
        assert date.floor_month(date.ymd('2021-01-01')) == date.ymd('2021-01-01')
        assert date.floor_month(date.ymd('2021-01-31')) == date.ymd('2021-01-01')
        assert date.floor_month(date.ymd('2021-12-01')) == date.ymd('2021-12-01')
        assert date.floor_month(date.ymd('2021-12-31')) == date.ymd('2021-12-01')

    def test_floor_quarter(self):
        # default argument first_fiscal_month of 1
        assert date.floor_quarter(date.ymd('2021-01-01')) == date.ymd('2021-01-01')
        assert date.floor_quarter(date.ymd('2021-01-31')) == date.ymd('2021-01-01')
        assert date.floor_quarter(date.ymd('2021-02-01')) == date.ymd('2021-01-01')
        assert date.floor_quarter(date.ymd('2021-02-28')) == date.ymd('2021-01-01')
        assert date.floor_quarter(date.ymd('2021-03-01')) == date.ymd('2021-01-01')
        assert date.floor_quarter(date.ymd('2021-03-31')) == date.ymd('2021-01-01')
        assert date.floor_quarter(date.ymd('2021-04-01')) == date.ymd('2021-04-01')
        assert date.floor_quarter(date.ymd('2021-04-30')) == date.ymd('2021-04-01')
        assert date.floor_quarter(date.ymd('2021-05-01')) == date.ymd('2021-04-01')
        assert date.floor_quarter(date.ymd('2021-05-31')) == date.ymd('2021-04-01')
        assert date.floor_quarter(date.ymd('2021-06-01')) == date.ymd('2021-04-01')
        assert date.floor_quarter(date.ymd('2021-06-30')) == date.ymd('2021-04-01')
        assert date.floor_quarter(date.ymd('2021-07-01')) == date.ymd('2021-07-01')
        assert date.floor_quarter(date.ymd('2021-07-31')) == date.ymd('2021-07-01')
        assert date.floor_quarter(date.ymd('2021-08-01')) == date.ymd('2021-07-01')
        assert date.floor_quarter(date.ymd('2021-08-31')) == date.ymd('2021-07-01')
        assert date.floor_quarter(date.ymd('2021-09-01')) == date.ymd('2021-07-01')
        assert date.floor_quarter(date.ymd('2021-09-30')) == date.ymd('2021-07-01')
        assert date.floor_quarter(date.ymd('2021-10-01')) == date.ymd('2021-10-01')
        assert date.floor_quarter(date.ymd('2021-10-31')) == date.ymd('2021-10-01')
        assert date.floor_quarter(date.ymd('2021-11-01')) == date.ymd('2021-10-01')
        assert date.floor_quarter(date.ymd('2021-11-30')) == date.ymd('2021-10-01')
        assert date.floor_quarter(date.ymd('2021-12-01')) == date.ymd('2021-10-01')
        assert date.floor_quarter(date.ymd('2021-12-31')) == date.ymd('2021-10-01')

        # fiscal quarter starts in November
        assert date.floor_quarter(date.ymd('2021-01-01'), first_fiscal_month=11) == date.ymd('2020-11-01')
        assert date.floor_quarter(date.ymd('2021-01-31'), first_fiscal_month=11) == date.ymd('2020-11-01')
        assert date.floor_quarter(date.ymd('2021-02-01'), first_fiscal_month=11) == date.ymd('2021-02-01')
        assert date.floor_quarter(date.ymd('2021-02-28'), first_fiscal_month=11) == date.ymd('2021-02-01')
        assert date.floor_quarter(date.ymd('2021-03-01'), first_fiscal_month=11) == date.ymd('2021-02-01')
        assert date.floor_quarter(date.ymd('2021-03-31'), first_fiscal_month=11) == date.ymd('2021-02-01')
        assert date.floor_quarter(date.ymd('2021-04-01'), first_fiscal_month=11) == date.ymd('2021-02-01')
        assert date.floor_quarter(date.ymd('2021-04-30'), first_fiscal_month=11) == date.ymd('2021-02-01')
        assert date.floor_quarter(date.ymd('2021-05-01'), first_fiscal_month=11) == date.ymd('2021-05-01')
        assert date.floor_quarter(date.ymd('2021-05-31'), first_fiscal_month=11) == date.ymd('2021-05-01')
        assert date.floor_quarter(date.ymd('2021-06-01'), first_fiscal_month=11) == date.ymd('2021-05-01')
        assert date.floor_quarter(date.ymd('2021-06-30'), first_fiscal_month=11) == date.ymd('2021-05-01')
        assert date.floor_quarter(date.ymd('2021-07-01'), first_fiscal_month=11) == date.ymd('2021-05-01')
        assert date.floor_quarter(date.ymd('2021-07-31'), first_fiscal_month=11) == date.ymd('2021-05-01')
        assert date.floor_quarter(date.ymd('2021-08-01'), first_fiscal_month=11) == date.ymd('2021-08-01')
        assert date.floor_quarter(date.ymd('2021-08-31'), first_fiscal_month=11) == date.ymd('2021-08-01')
        assert date.floor_quarter(date.ymd('2021-09-01'), first_fiscal_month=11) == date.ymd('2021-08-01')
        assert date.floor_quarter(date.ymd('2021-09-30'), first_fiscal_month=11) == date.ymd('2021-08-01')
        assert date.floor_quarter(date.ymd('2021-10-01'), first_fiscal_month=11) == date.ymd('2021-08-01')
        assert date.floor_quarter(date.ymd('2021-10-31'), first_fiscal_month=11) == date.ymd('2021-08-01')
        assert date.floor_quarter(date.ymd('2021-11-01'), first_fiscal_month=11) == date.ymd('2021-11-01')
        assert date.floor_quarter(date.ymd('2021-11-30'), first_fiscal_month=11) == date.ymd('2021-11-01')
        assert date.floor_quarter(date.ymd('2021-12-01'), first_fiscal_month=11) == date.ymd('2021-11-01')
        assert date.floor_quarter(date.ymd('2021-12-31'), first_fiscal_month=11) == date.ymd('2021-11-01')

        # fiscal quarter starts in June
        assert date.floor_quarter(date.ymd('2021-01-01'), first_fiscal_month=6) == date.ymd('2020-12-01')
        assert date.floor_quarter(date.ymd('2021-01-31'), first_fiscal_month=6) == date.ymd('2020-12-01')
        assert date.floor_quarter(date.ymd('2021-02-01'), first_fiscal_month=6) == date.ymd('2020-12-01')
        assert date.floor_quarter(date.ymd('2021-02-28'), first_fiscal_month=6) == date.ymd('2020-12-01')
        assert date.floor_quarter(date.ymd('2021-03-01'), first_fiscal_month=6) == date.ymd('2021-03-01')
        assert date.floor_quarter(date.ymd('2021-03-31'), first_fiscal_month=6) == date.ymd('2021-03-01')
        assert date.floor_quarter(date.ymd('2021-04-01'), first_fiscal_month=6) == date.ymd('2021-03-01')
        assert date.floor_quarter(date.ymd('2021-04-30'), first_fiscal_month=6) == date.ymd('2021-03-01')
        assert date.floor_quarter(date.ymd('2021-05-01'), first_fiscal_month=6) == date.ymd('2021-03-01')
        assert date.floor_quarter(date.ymd('2021-05-31'), first_fiscal_month=6) == date.ymd('2021-03-01')
        assert date.floor_quarter(date.ymd('2021-06-01'), first_fiscal_month=6) == date.ymd('2021-06-01')
        assert date.floor_quarter(date.ymd('2021-06-30'), first_fiscal_month=6) == date.ymd('2021-06-01')
        assert date.floor_quarter(date.ymd('2021-07-01'), first_fiscal_month=6) == date.ymd('2021-06-01')
        assert date.floor_quarter(date.ymd('2021-07-31'), first_fiscal_month=6) == date.ymd('2021-06-01')
        assert date.floor_quarter(date.ymd('2021-08-01'), first_fiscal_month=6) == date.ymd('2021-06-01')
        assert date.floor_quarter(date.ymd('2021-08-31'), first_fiscal_month=6) == date.ymd('2021-06-01')
        assert date.floor_quarter(date.ymd('2021-09-01'), first_fiscal_month=6) == date.ymd('2021-09-01')
        assert date.floor_quarter(date.ymd('2021-09-30'), first_fiscal_month=6) == date.ymd('2021-09-01')
        assert date.floor_quarter(date.ymd('2021-10-01'), first_fiscal_month=6) == date.ymd('2021-09-01')
        assert date.floor_quarter(date.ymd('2021-10-31'), first_fiscal_month=6) == date.ymd('2021-09-01')
        assert date.floor_quarter(date.ymd('2021-11-01'), first_fiscal_month=6) == date.ymd('2021-09-01')
        assert date.floor_quarter(date.ymd('2021-11-30'), first_fiscal_month=6) == date.ymd('2021-09-01')
        assert date.floor_quarter(date.ymd('2021-12-01'), first_fiscal_month=6) == date.ymd('2021-12-01')
        assert date.floor_quarter(date.ymd('2021-12-31'), first_fiscal_month=6) == date.ymd('2021-12-01')


    def etc(self):

        import datetime
        datetime.datetime.strptime(string, "%Y-%m-%d").date()

        def ymd(date_string: str) -> datetime.date:
            return datetime.datetime.strptime(date_string, "%Y-%m-%d").date()

        ymd('2021-01-01')
        ymd('2021-01-30')

        def floor_month(date: datetime.date) -> datetime.date:
            return date.replace(day=1)

        floor_month(ymd('2021-01-30'))
        floor_month(ymd('2021-03-30'))

        from dateutil import relativedelta

        date = ymd('2021-01-31')


        def floor_quarter(date: datetime.date, first_fiscal_month: int = 1) -> datetime.date:
            """
            "Rounds" the date down (i.e. floor) to the start of the current quarter. 

            For example, if `date` is `2021-03-20` then `floor_quarter` will return `2021-01-01`.

            If the fiscal year starts on November (`first_fiscal_month is `11`; i.e. the quarter is November,
            December, January) and the date is `2022-01-28` then `floor_quarter` will return `2021-11-01`.

            Parameters
            ----------
            date : datetime.date
                a date

            first_fiscal_month: int
                the month index (e.g. Jan is 1, Feb, 2, etc.) that corresponds to the first month of the
                fiscal year.

            Returns
            -------
            date - the start of the current quarter
            """
            relative_start_index = ((first_fiscal_month - 1) % 3) + 1
            current_month_index = ((date.month - 1) % 3) + 1
            months_to_subtract = (((relative_start_index * -1) + current_month_index) % 3)

            return floor_month(date) - relativedelta.relativedelta(months=months_to_subtract)

        

        date = ymd('2021-01-31')
        date = ymd('2021-04-28')

        def date_to_string(date: datetime.date) -> str:
            return date.strftime("%Y-%m-%d")

        date_to_string(ymd('2021-01-31'))


        def format_as_month(date: datetime.date) -> str:
            return date.strftime("%Y-%b")

        format_as_month(ymd('2021-01-17'))

        date = ymd('2021-01-04')
        date.strftime("%Y-%Q")

        [1, 2] - 1


        pd.Timestamp(ymd('2021-01-14')).quarter


