from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Provides python helper functions and classes'
LONG_DESCRIPTION = 'Provides python helper functions and classes'

# Setting up
setup(
       # the name must match the folder name 'kercheval'
        name="kercheval", 
        version=VERSION,
        author="Shane Kercheval",
        author_email="shane.kercheval@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url='https://github.com/shane-kercheval/python-helpers',
        license='MIT',
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        keywords=['utilities'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
        ]
)
