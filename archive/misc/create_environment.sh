ENV_FOLDER=.env_temp

# create virtual environment
rm -rf $ENV_FOLDER
python3.9 -m venv $ENV_FOLDER
source $ENV_FOLDER/bin/activate

# install scikit-learn on Apple M1
brew install openblas
export OPENBLAS=$(/opt/homebrew/bin/brew --prefix openblas)
export CFLAGS="-falign-functions=8 ${CFLAGS}"
pip install scikit-learn

####
pip install pandas


# install psycopg2
# https://stackoverflow.com/questions/33866695/error-installing-psycopg2-on-macos-10-9-5?answertab=votes#tab-top
brew install postgresql
pip install psycopg2

# snowflake
pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v2.7.4/tested_requirements/requirements_39.reqs
pip install snowflake-connector-python==v2.7.4

pip install -U Jinja2

pip install -U matplotlib
pip install seaborn
pip install plotly
pip install plotly-express

pip install PyYAML

pip install scikit-optimize

# https://xgboost.readthedocs.io/en/latest/install.html
# pip install xgboost
brew install libomp
conda update -n base -c defaults conda
conda install -c conda-forge py-xgboost


pip freeze
