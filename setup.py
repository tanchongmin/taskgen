from setuptools import setup, find_packages

setup(
    name="taskgen",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "openai==1.3.6",
    ],
)
