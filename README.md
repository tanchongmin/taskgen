# TaskGen v0.0.1
### A Task-based agentic framework building on StrictJSON outputs by LLM agents
- Related Repositories: StrictJSON (https://github.com/tanchongmin/strictjson)

### TaskGen functionalities (see Agent.ipynb)
- Task-based Agents which will break down tasks into subtasks and solve them in bite-sized portions
- Agents with registered functions as skills

### Upcoming Agent Functionalities (coming soon!)
- Multiple agents in a Task Group
- Retrieval Augmented Generation (RAG) - based selection of functions (to be added)
- RAG-based selection of memory of few-shot examples of how to use functions and how to perform task based on similar tasks done in the past (to be added)

### Benefits of JSON messaging over agentic frameworks using conversational free-text like AutoGen
- JSON format helps do Chain-of-Thought prompting naturally and is less verbose than free text
- JSON format allows natural parsing of multiple output fields by agents
- StrictJSON helps to ensure all output fields are there and of the right format required for downstream processing

### Tutorials and Community Support
- Created: 17 Feb 2024
- Collaborators welcome
- Discussion Channel (my discord - John's AI Group): [discord.gg/bzp87AHJy5](discord.gg/bzp87AHJy5)

## How do I use this? 
1. Download package via command line ```pip install taskgen-ai```
2. Set up your OpenAPI API Key. Refer to ```Agent.ipynb``` for how to do it for Jupyter Notebooks.
3. Import the required functions from ```taskgen``` and use them!

# Agent Basics
- Create an agent by entering your agent's name and description
- Agents are task-based, so they will help generate subtasks to fulfil your main task

- Agents are made to be non-verbose, so they will just focus only on task instruction
    
## Example Agent Creation
```python
my_agent = Agent('Helpful assistant', 'You are a generalist agent')
```

## Example Agent Task Running - Split the assigned task into subtasks and execute each of them

```python
output = my_agent.run('Give me 5 words rhyming with cool, and make a 4-sentence poem using them')
```

`Subtask identified: Find 5 words that rhyme with 'cool'
Getting LLM to perform the following task: Find 5 words that rhyme with 'cool'
pool, rule, fool, tool, school`

`Subtask identified: Compose a 4-sentence poem using the words 'pool', 'rule', 'fool', 'tool', and 'school'
Getting LLM to perform the following task: Compose a 4-sentence poem using the words 'pool', 'rule', 'fool', 'tool', and 'school'
In the school, the golden rule is to never be a fool. Use your mind as a tool, and always follow the pool.`

`Task completed successfully!`

## Example Agent Reply to User - Reference the subtasks' output to answer the user's query
```python
print(my_agent.reply_user())
```

`
Here are 5 words that rhyme with "cool": pool, rule, fool, tool, school. Here is a 4-sentence poem using these words: "In the school, the golden rule is to never be a fool. Use your mind as a tool, and always follow the pool."
`

## Check Agent's Status
```python
my_agent.status()
```

`Agent Name: Helpful assistant`

`Agent Description: You are a generalist agent`

`Available Functions: ['use_llm', 'end_task']`

`Task: Give me 5 words rhyming with cool, and make a 4-sentence poem using them`

`Subtasks Completed:`

`Subtask: Find 5 words that rhyme with 'cool'`

`pool, rule, fool, tool, school`

`Subtask: Compose a 4-sentence poem using the words 'pool', 'rule', 'fool', 'tool', and 'school'`

`In the school, the golden rule is to never be a fool. Use your mind as a tool, and always follow the pool.`

`Is Task Completed: True`

# Functions
- Provides a function-like interface for repeated use of modular LLM-based functions (or wraps external functions for use with TaskGen)
- Use angle brackets <> to enclose input variable names. First input variable name to appear in `fn_description` will be first input variable and second to appear will be second input variable. For example, `fn_description = 'Adds up two numbers, <var1> and <var2>'` will result in a function with first input variable `var1` and second input variable `var2`
- (Optional) If you would like greater specificity in your function's input, you can describe the variable after the : in the input variable name, e.g. `<var1: an integer from 10 to 30`. Here, `var1` is the input variable and `an integer from 10 to 30` is the description.
- Inputs (compulsory):
    - **fn_description**: String. Function description to describe process of transforming input variables to output variables. Variables must be enclosed in <> and listed in order of appearance in function input.
    - **output_format**: String. Dictionary containing output variables names and description for each variable. Refer to StrictJSON-Overview.ipynb for details on type checking for `output_format`
    
- Inputs (optional):
    - **examples** - Dict or List[Dict]. Examples in Dictionary form with the input and output variables (list if more than one)
    - **external_fn** - Python Function. If defined, instead of using LLM to process the function, we will run the external function. 
        If there are multiple outputs of this function, we will map it to the keys of `output_format` in a one-to-one fashion
    - **fn_name** - String. If provided, this will be the name of the function. Ohterwise, if `external_fn` is provided, it will be the name of `external_fn`. Otherwise, we will use LLM to generate a function name from the `fn_description`
    - **kwargs** - Dict. Additional arguments you would like to pass on to the strict_json function
        
- Outputs:
    JSON of output variables in a dictionary
    
#### Example Internal LLM-Based Function
```python
# Construct the function: var1 will be first input variable, var2 will be second input variable and so on
sentence_style = Function(fn_description = 'Output a sentence with words <var1> and <var2> in the style of <var3>', 
                     output_format = {'output': 'sentence'})

# Use the function
sentence_style('ball', 'dog', 'happy') #var1, var2, var3
```

#### Example Output
```{'output': 'The happy dog chased the ball.'}```
    
#### Example External Function
```python
def binary_to_decimal(x):
    return int(str(x), 2)

# an external function with a single output variable, with an expressive variable description
b2d = Function(fn_description = 'Convert input <x: a binary number in base 2> to base 10', 
            output_format = {'output1': 'x in base 10'},
            external_fn = binary_to_decimal)

# Use the function
b2d(10) #x
```

#### Example Output
```{'output1': 2}```

# Power Up your Agents - Bring in Functions (aka Tools)
- After creating your agent, use `assign_functions` to assign a list of functions (of class Function) to it
- Function names will be automatically inferred if not specified
- Proceed to run tasks by using `run()`

```python
my_agent = Agent('Helpful assistant', 'You are a generalist agent')

my_agent.assign_functions([sentence_style, b2d])

output = my_agent.run('Generate me a happy sentence with a number and a ball. The number is 1001 converted to decimal')
```

`Subtask identified: Convert the binary number 1001 to decimal`

`Calling function binary_to_decimal with parameters {'x': '1001'}`

`Output of binary_to_decimal with input {'x': '1001'}: {'output1': 9}`

`Subtask identified: Generate a happy sentence with the number 1001 and a ball`

`Calling function generate_sentence_with_emotion with parameters {'obj': 'number 1001', 'entity': 'ball', 'emotion': 'happy'}`

`Output of generate_sentence_with_emotion with input {'obj': 'number 1001', 'entity': 'ball', 'emotion': 'happy'}: {'output': 'I am overjoyed to have found number 1001 ball!'}`

`Task completed successfully!`

