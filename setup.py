from setuptools import setup, find_packages

setup(
    name="strictjson",
    version="3.0.2",
    packages=find_packages(),
    install_requires=[
        "openai==1.3.6",
    ],
)