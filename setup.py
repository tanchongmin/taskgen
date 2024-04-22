from setuptools import setup, find_packages

setup(
    name="taskgen",
    version="1.3.2",
    packages=find_packages(),
    install_requires=[
        "openai>=1.3.6",
        "dill>=0.3.7"
    ],
)
