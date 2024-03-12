# Web Browser Implementation
This example demonstrates implementing a web-browser agent inspired from the paper [Towards General Computer Control: A Multimodal Agent for Red Dead Redemption II as a Case Study](https://arxiv.org/abs/2403.03186).

## Introduction

## Features
The agent will be able to perform the following tasks:
- Take a screenshot of the current page
- Extract text from the current page
- JSON representation of the current page, detailing interactive elements
- Grounding Dino to the current page
- Clicking on the interactive elements
- Typing text into input fields
- Scrolling the page
- Navigating between interactive elements
- Extracting the URL of the current page

For more details, refer to the tasks list:
[Planning what to do](Tasks.md)

## Prerequisites
- Python 3.10
- taskgen library
- python-dotenv library
- OpenAI API key

## Setup
Install the required dependencies:

```bash
pip install -r requirements.txt
```
Create a `.env` file in the project root directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage
1. Run `python web_agent.py`

