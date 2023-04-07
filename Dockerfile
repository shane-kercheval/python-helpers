FROM python:3.11

WORKDIR /code
ENV PYTHONPATH "${PYTHONPATH}:/code"

RUN apt-get update -y && apt-get install zsh -y
RUN PATH="$PATH:/usr/bin/zsh"

RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade build
RUN python -m pip install --upgrade twine
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

RUN pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v3.0.2/tested_requirements/requirements_311.reqs
RUN pip install snowflake-connector-python==v3.0.2

# RUN pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v3.0.2/tested_requirements/requirements_310.reqs
# RUN pip install snowflake-connector-python==v3.0.2

# RUN pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v2.7.4/tested_requirements/requirements_39.reqs
# RUN pip install snowflake-connector-python==v2.7.4

COPY .pypirc /root/.pypirc
