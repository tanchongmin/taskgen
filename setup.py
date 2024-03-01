from setuptools import setup, find_packages

setup(
    name="taskgen",
    version="0.0.8",
    packages=find_packages(),
    install_requires=[
        "openai==1.3.6",
    ],
)
