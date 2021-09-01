import datetime
import unittest

from dateutil.parser import parse

from helpsk import date
from tests.helpers import subtests_expected_vs_actual


# noinspection PyMethodMayBeStatic
class TestDate(unittest.TestCase):

    def test_fiscal_quarter_date(self):
        date_values = ['2020-12-01', '2020-12-15', '2020-12-31',
                       '2021-01-01', '2021-01-15', '2021-01-31',
                       '2021-02-01', '2021-02-15', '2021-02-28',
                       '2021-03-01', '2021-03-15', '2021-03-31',
                       '2021-04-01', '2021-04-15', '2021-04-30',
                       '2021-05-01', '2021-05-15', '2021-05-31',
                       '2021-06-01', '2021-06-15', '2021-06-30',
                       '2021-07-01', '2021-07-15', '2021-07-31',
                       '2021-08-01', '2021-08-15', '2021-08-31',
                       '2021-09-01', '2021-09-15', '2021-09-30',
                       '2021-10-01', '2021-10-15', '2021-10-31',
                       '2021-11-01', '2021-11-15', '2021-11-30',
                       '2021-12-01', '2021-12-15', '2021-12-31',
                       '2022-01-01', '2022-01-15', '2022-01-31']

        test_parameters = dict(include_year=True, fiscal_start=1)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [2020.4, 2020.4, 2020.4,
                    2021.1, 2021.1, 2021.1,
                    2021.1, 2021.1, 2021.1,
                    2021.1, 2021.1, 2021.1,
                    2021.2, 2021.2, 2021.2,
                    2021.2, 2021.2, 2021.2,
                    2021.2, 2021.2, 2021.2,
                    2021.3, 2021.3, 2021.3,
                    2021.3, 2021.3, 2021.3,
                    2021.3, 2021.3, 2021.3,
                    2021.4, 2021.4, 2021.4,
                    2021.4, 2021.4, 2021.4,
                    2021.4, 2021.4, 2021.4,
                    2022.1, 2022.1, 2022.1]

        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=False, fiscal_start=1)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [4, 4, 4,
                    1, 1, 1,
                    1, 1, 1,
                    1, 1, 1,
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                    3, 3, 3,
                    3, 3, 3,
                    3, 3, 3,
                    4, 4, 4,
                    4, 4, 4,
                    4, 4, 4,
                    1, 1, 1]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=True, fiscal_start=2)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [2021.4, 2021.4, 2021.4,
                    2021.4, 2021.4, 2021.4,
                    2022.1, 2022.1, 2022.1,
                    2022.1, 2022.1, 2022.1,
                    2022.1, 2022.1, 2022.1,
                    2022.2, 2022.2, 2022.2,
                    2022.2, 2022.2, 2022.2,
                    2022.2, 2022.2, 2022.2,
                    2022.3, 2022.3, 2022.3,
                    2022.3, 2022.3, 2022.3,
                    2022.3, 2022.3, 2022.3,
                    2022.4, 2022.4, 2022.4,
                    2022.4, 2022.4, 2022.4,
                    2022.4, 2022.4, 2022.4]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=False, fiscal_start=2)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [4, 4, 4,
                    4, 4, 4,
                    1, 1, 1,
                    1, 1, 1,
                    1, 1, 1,
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                    3, 3, 3,
                    3, 3, 3,
                    3, 3, 3,
                    4, 4, 4,
                    4, 4, 4,
                    4, 4, 4]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=True, fiscal_start=12)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [2021.1, 2021.1, 2021.1,  # 2020-Dec
                    2021.1, 2021.1, 2021.1,  # 2021-Jan
                    2021.1, 2021.1, 2021.1,  # 2021-Feb
                    2021.2, 2021.2, 2021.2,  # 2021-Mar
                    2021.2, 2021.2, 2021.2,  # 2021-Apr
                    2021.2, 2021.2, 2021.2,  # 2021-May
                    2021.3, 2021.3, 2021.3,  # 2021-Jun
                    2021.3, 2021.3, 2021.3,  # 2021-Jul
                    2021.3, 2021.3, 2021.3,  # 2021-Aug
                    2021.4, 2021.4, 2021.4,  # 2021-Sep
                    2021.4, 2021.4, 2021.4,  # 2021-Oct
                    2021.4, 2021.4, 2021.4,  # 2021-Nov
                    2022.1, 2022.1, 2022.1,  # 2021-Dec
                    2022.1, 2022.1, 2022.1]  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=False, fiscal_start=12)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [1, 1, 1,  # 2020-Dec
                    1, 1, 1,  # 2021-Jan
                    1, 1, 1,  # 2021-Feb
                    2, 2, 2,  # 2021-Mar
                    2, 2, 2,  # 2021-Apr
                    2, 2, 2,  # 2021-May
                    3, 3, 3,  # 2021-Jun
                    3, 3, 3,  # 2021-Jul
                    3, 3, 3,  # 2021-Aug
                    4, 4, 4,  # 2021-Sep
                    4, 4, 4,  # 2021-Oct
                    4, 4, 4,  # 2021-Nov
                    1, 1, 1,  # 2021-Dec
                    1, 1, 1]  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

    def test_fiscal_quarter_datetime(self):
        date_values = ['2020-12-01', '2020-12-15', '2020-12-31',
                       '2021-01-01', '2021-01-15', '2021-01-31',
                       '2021-02-01', '2021-02-15', '2021-02-28',
                       '2021-03-01', '2021-03-15', '2021-03-31',
                       '2021-04-01', '2021-04-15', '2021-04-30',
                       '2021-05-01', '2021-05-15', '2021-05-31',
                       '2021-06-01', '2021-06-15', '2021-06-30',
                       '2021-07-01', '2021-07-15', '2021-07-31',
                       '2021-08-01', '2021-08-15', '2021-08-31',
                       '2021-09-01', '2021-09-15', '2021-09-30',
                       '2021-10-01', '2021-10-15', '2021-10-31',
                       '2021-11-01', '2021-11-15', '2021-11-30',
                       '2021-12-01', '2021-12-15', '2021-12-31',
                       '2022-01-01', '2022-01-15', '2022-01-31']

        test_parameters = dict(include_year=True, fiscal_start=1)
        results = [date.fiscal_quarter(value=parse(x + ' 23:59:59'), **test_parameters) for x in date_values]
        expected = [2020.4, 2020.4, 2020.4,
                    2021.1, 2021.1, 2021.1,
                    2021.1, 2021.1, 2021.1,
                    2021.1, 2021.1, 2021.1,
                    2021.2, 2021.2, 2021.2,
                    2021.2, 2021.2, 2021.2,
                    2021.2, 2021.2, 2021.2,
                    2021.3, 2021.3, 2021.3,
                    2021.3, 2021.3, 2021.3,
                    2021.3, 2021.3, 2021.3,
                    2021.4, 2021.4, 2021.4,
                    2021.4, 2021.4, 2021.4,
                    2021.4, 2021.4, 2021.4,
                    2022.1, 2022.1, 2022.1]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=False, fiscal_start=1)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [4, 4, 4,
                    1, 1, 1,
                    1, 1, 1,
                    1, 1, 1,
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                    3, 3, 3,
                    3, 3, 3,
                    3, 3, 3,
                    4, 4, 4,
                    4, 4, 4,
                    4, 4, 4,
                    1, 1, 1]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=True, fiscal_start=2)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [2021.4, 2021.4, 2021.4,
                    2021.4, 2021.4, 2021.4,
                    2022.1, 2022.1, 2022.1,
                    2022.1, 2022.1, 2022.1,
                    2022.1, 2022.1, 2022.1,
                    2022.2, 2022.2, 2022.2,
                    2022.2, 2022.2, 2022.2,
                    2022.2, 2022.2, 2022.2,
                    2022.3, 2022.3, 2022.3,
                    2022.3, 2022.3, 2022.3,
                    2022.3, 2022.3, 2022.3,
                    2022.4, 2022.4, 2022.4,
                    2022.4, 2022.4, 2022.4,
                    2022.4, 2022.4, 2022.4]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=False, fiscal_start=2)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [4, 4, 4,
                    4, 4, 4,
                    1, 1, 1,
                    1, 1, 1,
                    1, 1, 1,
                    2, 2, 2,
                    2, 2, 2,
                    2, 2, 2,
                    3, 3, 3,
                    3, 3, 3,
                    3, 3, 3,
                    4, 4, 4,
                    4, 4, 4,
                    4, 4, 4]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=True, fiscal_start=12)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [2021.1, 2021.1, 2021.1,  # 2020-Dec
                    2021.1, 2021.1, 2021.1,  # 2021-Jan
                    2021.1, 2021.1, 2021.1,  # 2021-Feb
                    2021.2, 2021.2, 2021.2,  # 2021-Mar
                    2021.2, 2021.2, 2021.2,  # 2021-Apr
                    2021.2, 2021.2, 2021.2,  # 2021-May
                    2021.3, 2021.3, 2021.3,  # 2021-Jun
                    2021.3, 2021.3, 2021.3,  # 2021-Jul
                    2021.3, 2021.3, 2021.3,  # 2021-Aug
                    2021.4, 2021.4, 2021.4,  # 2021-Sep
                    2021.4, 2021.4, 2021.4,  # 2021-Oct
                    2021.4, 2021.4, 2021.4,  # 2021-Nov
                    2022.1, 2022.1, 2022.1,  # 2021-Dec
                    2022.1, 2022.1, 2022.1]  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(include_year=False, fiscal_start=12)
        results = [date.fiscal_quarter(value=parse(x), **test_parameters) for x in date_values]
        expected = [1, 1, 1,  # 2020-Dec
                    1, 1, 1,  # 2021-Jan
                    1, 1, 1,  # 2021-Feb
                    2, 2, 2,  # 2021-Mar
                    2, 2, 2,  # 2021-Apr
                    2, 2, 2,  # 2021-May
                    3, 3, 3,  # 2021-Jun
                    3, 3, 3,  # 2021-Jul
                    3, 3, 3,  # 2021-Aug
                    4, 4, 4,  # 2021-Sep
                    4, 4, 4,  # 2021-Oct
                    4, 4, 4,  # 2021-Nov
                    1, 1, 1,  # 2021-Dec
                    1, 1, 1]  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

    def test_to_string_date(self):
        date_values = ['2020-12-01', '2020-12-15', '2020-12-31',
                       '2021-01-01', '2021-01-15', '2021-01-31',
                       '2021-02-01', '2021-02-15', '2021-02-28',
                       '2021-03-01', '2021-03-15', '2021-03-31',
                       '2021-04-01', '2021-04-15', '2021-04-30',
                       '2021-05-01', '2021-05-15', '2021-05-31',
                       '2021-06-01', '2021-06-15', '2021-06-30',
                       '2021-07-01', '2021-07-15', '2021-07-31',
                       '2021-08-01', '2021-08-15', '2021-08-31',
                       '2021-09-01', '2021-09-15', '2021-09-30',
                       '2021-10-01', '2021-10-15', '2021-10-31',
                       '2021-11-01', '2021-11-15', '2021-11-30',
                       '2021-12-01', '2021-12-15', '2021-12-31',
                       '2022-01-01', '2022-01-15', '2022-01-31']

        test_parameters = dict(granularity=date.Granularity.DAY)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=date_values,
                                    **test_parameters)

        test_parameters = dict(granularity=date.Granularity.MONTH)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        expected = ['2020-Dec', '2020-Dec', '2020-Dec',
                    '2021-Jan', '2021-Jan', '2021-Jan',
                    '2021-Feb', '2021-Feb', '2021-Feb',
                    '2021-Mar', '2021-Mar', '2021-Mar',
                    '2021-Apr', '2021-Apr', '2021-Apr',
                    '2021-May', '2021-May', '2021-May',
                    '2021-Jun', '2021-Jun', '2021-Jun',
                    '2021-Jul', '2021-Jul', '2021-Jul',
                    '2021-Aug', '2021-Aug', '2021-Aug',
                    '2021-Sep', '2021-Sep', '2021-Sep',
                    '2021-Oct', '2021-Oct', '2021-Oct',
                    '2021-Nov', '2021-Nov', '2021-Nov',
                    '2021-Dec', '2021-Dec', '2021-Dec',
                    '2022-Jan', '2022-Jan', '2022-Jan']
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(granularity=date.Granularity.QUARTER, fiscal_start=1)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        expected = ['2020-Q4', '2020-Q4', '2020-Q4',  # 2020-Dec
                    '2021-Q1', '2021-Q1', '2021-Q1',  # 2021-Jan
                    '2021-Q1', '2021-Q1', '2021-Q1',  # 2021-Feb
                    '2021-Q1', '2021-Q1', '2021-Q1',  # 2021-Mar
                    '2021-Q2', '2021-Q2', '2021-Q2',  # 2021-Apr
                    '2021-Q2', '2021-Q2', '2021-Q2',  # 2021-May
                    '2021-Q2', '2021-Q2', '2021-Q2',  # 2021-Jun
                    '2021-Q3', '2021-Q3', '2021-Q3',  # 2021-Jul
                    '2021-Q3', '2021-Q3', '2021-Q3',  # 2021-Aug
                    '2021-Q3', '2021-Q3', '2021-Q3',  # 2021-Sep
                    '2021-Q4', '2021-Q4', '2021-Q4',  # 2021-Oct
                    '2021-Q4', '2021-Q4', '2021-Q4',  # 2021-Nov
                    '2021-Q4', '2021-Q4', '2021-Q4',  # 2021-Dec
                    '2022-Q1', '2022-Q1', '2022-Q1']  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(granularity=date.Granularity.QUARTER, fiscal_start=2)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        expected = ['2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2020-Dec
                    '2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2021-Jan
                    '2022-FQ1', '2022-FQ1', '2022-FQ1',  # 2021-Feb
                    '2022-FQ1', '2022-FQ1', '2022-FQ1',  # 2021-Mar
                    '2022-FQ1', '2022-FQ1', '2022-FQ1',  # 2021-Apr
                    '2022-FQ2', '2022-FQ2', '2022-FQ2',  # 2021-May
                    '2022-FQ2', '2022-FQ2', '2022-FQ2',  # 2021-Jun
                    '2022-FQ2', '2022-FQ2', '2022-FQ2',  # 2021-Jul
                    '2022-FQ3', '2022-FQ3', '2022-FQ3',  # 2021-Aug
                    '2022-FQ3', '2022-FQ3', '2022-FQ3',  # 2021-Sep
                    '2022-FQ3', '2022-FQ3', '2022-FQ3',  # 2021-Oct
                    '2022-FQ4', '2022-FQ4', '2022-FQ4',  # 2021-Nov
                    '2022-FQ4', '2022-FQ4', '2022-FQ4',  # 2021-Dec
                    '2022-FQ4', '2022-FQ4', '2022-FQ4']  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(granularity=date.Granularity.QUARTER, fiscal_start=12)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        expected = ['2021-FQ1', '2021-FQ1', '2021-FQ1',  # 2020-Dec
                    '2021-FQ1', '2021-FQ1', '2021-FQ1',  # 2021-Jan
                    '2021-FQ1', '2021-FQ1', '2021-FQ1',  # 2021-Feb
                    '2021-FQ2', '2021-FQ2', '2021-FQ2',  # 2021-Mar
                    '2021-FQ2', '2021-FQ2', '2021-FQ2',  # 2021-Apr
                    '2021-FQ2', '2021-FQ2', '2021-FQ2',  # 2021-May
                    '2021-FQ3', '2021-FQ3', '2021-FQ3',  # 2021-Jun
                    '2021-FQ3', '2021-FQ3', '2021-FQ3',  # 2021-Jul
                    '2021-FQ3', '2021-FQ3', '2021-FQ3',  # 2021-Aug
                    '2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2021-Sep
                    '2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2021-Oct
                    '2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2021-Nov
                    '2022-FQ1', '2022-FQ1', '2022-FQ1',  # 2021-Dec
                    '2022-FQ1', '2022-FQ1', '2022-FQ1']  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

    def test_to_string_datetime(self):
        date_values = ['2020-12-01', '2020-12-15', '2020-12-31',
                       '2021-01-01', '2021-01-15', '2021-01-31',
                       '2021-02-01', '2021-02-15', '2021-02-28',
                       '2021-03-01', '2021-03-15', '2021-03-31',
                       '2021-04-01', '2021-04-15', '2021-04-30',
                       '2021-05-01', '2021-05-15', '2021-05-31',
                       '2021-06-01', '2021-06-15', '2021-06-30',
                       '2021-07-01', '2021-07-15', '2021-07-31',
                       '2021-08-01', '2021-08-15', '2021-08-31',
                       '2021-09-01', '2021-09-15', '2021-09-30',
                       '2021-10-01', '2021-10-15', '2021-10-31',
                       '2021-11-01', '2021-11-15', '2021-11-30',
                       '2021-12-01', '2021-12-15', '2021-12-31',
                       '2022-01-01', '2022-01-15', '2022-01-31']

        test_parameters = dict(granularity=date.Granularity.DAY)
        results = [date.to_string(value=parse(x + ' 23:59:59'), **test_parameters) for x in date_values]
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=date_values,
                                    **test_parameters)

        test_parameters = dict(granularity=date.Granularity.MONTH)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        expected = ['2020-Dec', '2020-Dec', '2020-Dec',
                    '2021-Jan', '2021-Jan', '2021-Jan',
                    '2021-Feb', '2021-Feb', '2021-Feb',
                    '2021-Mar', '2021-Mar', '2021-Mar',
                    '2021-Apr', '2021-Apr', '2021-Apr',
                    '2021-May', '2021-May', '2021-May',
                    '2021-Jun', '2021-Jun', '2021-Jun',
                    '2021-Jul', '2021-Jul', '2021-Jul',
                    '2021-Aug', '2021-Aug', '2021-Aug',
                    '2021-Sep', '2021-Sep', '2021-Sep',
                    '2021-Oct', '2021-Oct', '2021-Oct',
                    '2021-Nov', '2021-Nov', '2021-Nov',
                    '2021-Dec', '2021-Dec', '2021-Dec',
                    '2022-Jan', '2022-Jan', '2022-Jan']
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(granularity=date.Granularity.QUARTER, fiscal_start=1)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        expected = ['2020-Q4', '2020-Q4', '2020-Q4',  # 2020-Dec
                    '2021-Q1', '2021-Q1', '2021-Q1',  # 2021-Jan
                    '2021-Q1', '2021-Q1', '2021-Q1',  # 2021-Feb
                    '2021-Q1', '2021-Q1', '2021-Q1',  # 2021-Mar
                    '2021-Q2', '2021-Q2', '2021-Q2',  # 2021-Apr
                    '2021-Q2', '2021-Q2', '2021-Q2',  # 2021-May
                    '2021-Q2', '2021-Q2', '2021-Q2',  # 2021-Jun
                    '2021-Q3', '2021-Q3', '2021-Q3',  # 2021-Jul
                    '2021-Q3', '2021-Q3', '2021-Q3',  # 2021-Aug
                    '2021-Q3', '2021-Q3', '2021-Q3',  # 2021-Sep
                    '2021-Q4', '2021-Q4', '2021-Q4',  # 2021-Oct
                    '2021-Q4', '2021-Q4', '2021-Q4',  # 2021-Nov
                    '2021-Q4', '2021-Q4', '2021-Q4',  # 2021-Dec
                    '2022-Q1', '2022-Q1', '2022-Q1']  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(granularity=date.Granularity.QUARTER, fiscal_start=2)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        expected = ['2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2020-Dec
                    '2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2021-Jan
                    '2022-FQ1', '2022-FQ1', '2022-FQ1',  # 2021-Feb
                    '2022-FQ1', '2022-FQ1', '2022-FQ1',  # 2021-Mar
                    '2022-FQ1', '2022-FQ1', '2022-FQ1',  # 2021-Apr
                    '2022-FQ2', '2022-FQ2', '2022-FQ2',  # 2021-May
                    '2022-FQ2', '2022-FQ2', '2022-FQ2',  # 2021-Jun
                    '2022-FQ2', '2022-FQ2', '2022-FQ2',  # 2021-Jul
                    '2022-FQ3', '2022-FQ3', '2022-FQ3',  # 2021-Aug
                    '2022-FQ3', '2022-FQ3', '2022-FQ3',  # 2021-Sep
                    '2022-FQ3', '2022-FQ3', '2022-FQ3',  # 2021-Oct
                    '2022-FQ4', '2022-FQ4', '2022-FQ4',  # 2021-Nov
                    '2022-FQ4', '2022-FQ4', '2022-FQ4',  # 2021-Dec
                    '2022-FQ4', '2022-FQ4', '2022-FQ4']  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

        test_parameters = dict(granularity=date.Granularity.QUARTER, fiscal_start=12)
        results = [date.to_string(value=parse(x), **test_parameters) for x in date_values]
        expected = ['2021-FQ1', '2021-FQ1', '2021-FQ1',  # 2020-Dec
                    '2021-FQ1', '2021-FQ1', '2021-FQ1',  # 2021-Jan
                    '2021-FQ1', '2021-FQ1', '2021-FQ1',  # 2021-Feb
                    '2021-FQ2', '2021-FQ2', '2021-FQ2',  # 2021-Mar
                    '2021-FQ2', '2021-FQ2', '2021-FQ2',  # 2021-Apr
                    '2021-FQ2', '2021-FQ2', '2021-FQ2',  # 2021-May
                    '2021-FQ3', '2021-FQ3', '2021-FQ3',  # 2021-Jun
                    '2021-FQ3', '2021-FQ3', '2021-FQ3',  # 2021-Jul
                    '2021-FQ3', '2021-FQ3', '2021-FQ3',  # 2021-Aug
                    '2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2021-Sep
                    '2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2021-Oct
                    '2021-FQ4', '2021-FQ4', '2021-FQ4',  # 2021-Nov
                    '2022-FQ1', '2022-FQ1', '2022-FQ1',  # 2021-Dec
                    '2022-FQ1', '2022-FQ1', '2022-FQ1']  # 2022-Jan
        subtests_expected_vs_actual(test_case=self, actual_values=results, expected_values=expected,
                                    **test_parameters)

    def test_floor_day(self):
        # test datetime
        value = datetime.datetime(year=2021, month=2, day=13, hour=23, minute=45, second=55)
        self.assertEqual(date.floor(value, granularity=date.Granularity.DAY),
                         parse('2021-02-13').date())
        self.assertEqual(date.floor(value),
                         parse('2021-02-13').date())
        # test date
        value = datetime.date(year=2021, month=2, day=13)
        self.assertEqual(date.floor(value, granularity=date.Granularity.DAY),
                         parse('2021-02-13').date())
        self.assertEqual(date.floor(value),
                         parse('2021-02-13').date())

    def test_floor_month(self):
        # test datetime
        value = datetime.datetime(year=2021, month=1, day=1, hour=23, minute=45, second=55)
        self.assertEqual(date.floor(value, granularity=date.Granularity.MONTH),
                         parse('2021-01-01').date())
        value = datetime.datetime(year=2021, month=1, day=31, hour=23, minute=45, second=55)
        self.assertEqual(date.floor(value, granularity=date.Granularity.MONTH),
                         parse('2021-01-01').date())
        value = datetime.datetime(year=2021, month=12, day=1, hour=23, minute=45, second=55)
        self.assertEqual(date.floor(value, granularity=date.Granularity.MONTH),
                         parse('2021-12-01').date())
        value = datetime.datetime(year=2021, month=12, day=31, hour=23, minute=45, second=55)
        self.assertEqual(date.floor(value, granularity=date.Granularity.MONTH),
                         parse('2021-12-01').date())

        # test date
        self.assertEqual(date.floor(parse('2021-01-01'), granularity=date.Granularity.MONTH),
                         parse('2021-01-01').date())
        self.assertEqual(date.floor(parse('2021-01-31'), granularity=date.Granularity.MONTH),
                         parse('2021-01-01').date())
        self.assertEqual(date.floor(parse('2021-12-01'), granularity=date.Granularity.MONTH),
                         parse('2021-12-01').date())
        self.assertEqual(date.floor(parse('2021-12-31'), granularity=date.Granularity.MONTH),
                         parse('2021-12-01').date())

    def test_floor_quarter(self):
        # default argument fiscal_start of 1
        self.assertEqual(date.floor(parse('2021-01-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-01-01').date())
        self.assertEqual(date.floor(parse('2021-01-31'), granularity=date.Granularity.QUARTER),
                         parse('2021-01-01').date())
        self.assertEqual(date.floor(parse('2021-02-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-01-01').date())
        self.assertEqual(date.floor(parse('2021-02-28'), granularity=date.Granularity.QUARTER),
                         parse('2021-01-01').date())
        self.assertEqual(date.floor(parse('2021-03-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-01-01').date())
        self.assertEqual(date.floor(parse('2021-03-31'), granularity=date.Granularity.QUARTER),
                         parse('2021-01-01').date())
        self.assertEqual(date.floor(parse('2021-04-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-04-01').date())
        self.assertEqual(date.floor(parse('2021-04-30'), granularity=date.Granularity.QUARTER),
                         parse('2021-04-01').date())
        self.assertEqual(date.floor(parse('2021-05-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-04-01').date())
        self.assertEqual(date.floor(parse('2021-05-31'), granularity=date.Granularity.QUARTER),
                         parse('2021-04-01').date())
        self.assertEqual(date.floor(parse('2021-06-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-04-01').date())
        self.assertEqual(date.floor(parse('2021-06-30'), granularity=date.Granularity.QUARTER),
                         parse('2021-04-01').date())
        self.assertEqual(date.floor(parse('2021-07-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-07-01').date())
        self.assertEqual(date.floor(parse('2021-07-31'), granularity=date.Granularity.QUARTER),
                         parse('2021-07-01').date())
        self.assertEqual(date.floor(parse('2021-08-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-07-01').date())
        self.assertEqual(date.floor(parse('2021-08-31'), granularity=date.Granularity.QUARTER),
                         parse('2021-07-01').date())
        self.assertEqual(date.floor(parse('2021-09-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-07-01').date())
        self.assertEqual(date.floor(parse('2021-09-30'), granularity=date.Granularity.QUARTER),
                         parse('2021-07-01').date())
        self.assertEqual(date.floor(parse('2021-10-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-10-01').date())
        self.assertEqual(date.floor(parse('2021-10-31'), granularity=date.Granularity.QUARTER),
                         parse('2021-10-01').date())
        self.assertEqual(date.floor(parse('2021-11-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-10-01').date())
        self.assertEqual(date.floor(parse('2021-11-30'), granularity=date.Granularity.QUARTER),
                         parse('2021-10-01').date())
        self.assertEqual(date.floor(parse('2021-12-01'), granularity=date.Granularity.QUARTER),
                         parse('2021-10-01').date())
        self.assertEqual(date.floor(parse('2021-12-31'), granularity=date.Granularity.QUARTER),
                         parse('2021-10-01').date())

        # fiscal quarter starts in February
        self.assertEqual(date.floor(parse('2021-01-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2020-11-01').date())
        self.assertEqual(date.floor(parse('2021-01-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2020-11-01').date())
        self.assertEqual(date.floor(parse('2021-02-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-02-28'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-03-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-03-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-04-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-04-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-05-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-05-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-06-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-06-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-07-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-07-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-08-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-08-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-09-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-09-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-10-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-10-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-11-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-11-01').date())
        self.assertEqual(date.floor(parse('2021-11-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-11-01').date())
        self.assertEqual(date.floor(parse('2021-12-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-11-01').date())
        self.assertEqual(date.floor(parse('2021-12-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=2),
                         parse('2021-11-01').date())

        # fiscal quarter starts in November (should be same as February)
        self.assertEqual(date.floor(parse('2021-01-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2020-11-01').date())
        self.assertEqual(date.floor(parse('2021-01-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2020-11-01').date())
        self.assertEqual(date.floor(parse('2021-02-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-02-28'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-03-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-03-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-04-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-04-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-02-01').date())
        self.assertEqual(date.floor(parse('2021-05-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-05-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-06-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-06-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-07-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-07-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-05-01').date())
        self.assertEqual(date.floor(parse('2021-08-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-08-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-09-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-09-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-10-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-10-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-08-01').date())
        self.assertEqual(date.floor(parse('2021-11-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-11-01').date())
        self.assertEqual(date.floor(parse('2021-11-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-11-01').date())
        self.assertEqual(date.floor(parse('2021-12-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-11-01').date())
        self.assertEqual(date.floor(parse('2021-12-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=11),
                         parse('2021-11-01').date())

        # fiscal quarter starts in June
        self.assertEqual(date.floor(parse('2021-01-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2020-12-01').date())
        self.assertEqual(date.floor(parse('2021-01-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2020-12-01').date())
        self.assertEqual(date.floor(parse('2021-02-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2020-12-01').date())
        self.assertEqual(date.floor(parse('2021-02-28'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2020-12-01').date())
        self.assertEqual(date.floor(parse('2021-03-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-03-01').date())
        self.assertEqual(date.floor(parse('2021-03-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-03-01').date())
        self.assertEqual(date.floor(parse('2021-04-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-03-01').date())
        self.assertEqual(date.floor(parse('2021-04-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-03-01').date())
        self.assertEqual(date.floor(parse('2021-05-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-03-01').date())
        self.assertEqual(date.floor(parse('2021-05-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-03-01').date())
        self.assertEqual(date.floor(parse('2021-06-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-06-01').date())
        self.assertEqual(date.floor(parse('2021-06-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-06-01').date())
        self.assertEqual(date.floor(parse('2021-07-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-06-01').date())
        self.assertEqual(date.floor(parse('2021-07-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-06-01').date())
        self.assertEqual(date.floor(parse('2021-08-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-06-01').date())
        self.assertEqual(date.floor(parse('2021-08-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-06-01').date())
        self.assertEqual(date.floor(parse('2021-09-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-09-01').date())
        self.assertEqual(date.floor(parse('2021-09-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-09-01').date())
        self.assertEqual(date.floor(parse('2021-10-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-09-01').date())
        self.assertEqual(date.floor(parse('2021-10-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-09-01').date())
        self.assertEqual(date.floor(parse('2021-11-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-09-01').date())
        self.assertEqual(date.floor(parse('2021-11-30'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-09-01').date())
        self.assertEqual(date.floor(parse('2021-12-01'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-12-01').date())
        self.assertEqual(date.floor(parse('2021-12-31'), granularity=date.Granularity.QUARTER,
                                    fiscal_start=6),
                         parse('2021-12-01').date())
