# TaskGen v1.0.0
### A Task-based agentic framework building on StrictJSON outputs by LLM agents
- Related Repositories: StrictJSON (https://github.com/tanchongmin/strictjson)

### Creator's Preamble
Happy to share that the task-based agentic framework I have been working on - TaskGen - is largely complete! 

Noteable features include:
- Splitting of Tasks into subtasks for bite-sized solutions for each subtask
- Single Agent with LLM Functions
- Single Agent with External Functions
- Meta Agent with Inner Agents as Functions
- Shared Variables for multi-modality support
- Retrieval Augmented Generation (RAG) over Function space, as well as to provide additional Context to task

I am quite sure that this is the best open-source agentic framework for task-based execution out there! 
Existing frameworks like AutoGen rely too much on conversational text which is lengthy and not targeted.
TaskGen uses StrictJSON (JSON parser with type checking and more!) as the core, and agents are efficient and are able to do Chain of Thought natively using JSON keys and descriptions as a guide.

What can you do to help: 
- Star the github so more people can use it (It's open source and free to use, even commercially!)
- Contribute your favourite external function integrations so that it can be much more boilerplate for others to use :)
- Contribute template Jupyter Notebooks for your favourite use cases :)

I can't wait to see what this new framework can do for you!

### Benefits of JSON messaging over agentic frameworks using conversational free-text like AutoGen
- JSON format helps do Chain-of-Thought prompting naturally and is less verbose than free text
- JSON format allows natural parsing of multiple output fields by agents
- StrictJSON helps to ensure all output fields are there and of the right format required for downstream processing

### Tutorials and Community Support
- Created: 17 Feb 2024 by John Tan Chong Min
- Collaborators welcome
- Discussion Channel (my discord - John's AI Group): [https://discord.gg/bzp87AHJy5](https://discord.gg/bzp87AHJy5)

## How do I use this? 
1. Download package via command line ```pip install taskgen-ai```
2. Set up your OpenAPI API Key
3. Import the required functions from ```taskgen``` and use them!

# Agent Basics
- Create an agent by entering your agent's name and description
- Agents are task-based, so they will help generate subtasks to fulfil your main task
- Agents are made to be non-verbose, so they will just focus only on task instruction (Much more efficient compared to conversational-based agentic frameworks like AutoGen)
    
## Example Agent Creation
```python
my_agent = Agent('Helpful assistant', 'You are a generalist agent')
```

## Example Agent Task Running - Split the assigned task into subtasks and execute each of them

```python
# Run your agent
output = my_agent.run('Give me 5 words rhyming with cool, and make a 4-sentence poem using them')
```

`Subtask identified: Find 5 words that rhyme with 'cool'`

`Getting LLM to perform the following task: Find 5 words that rhyme with 'cool'`
> pool, rule, fool, tool, school

`Subtask identified: Compose a 4-sentence poem using the words 'pool', 'rule', 'fool', 'tool', and 'school'`

`Getting LLM to perform the following task: Compose a 4-sentence poem using the words 'pool', 'rule', 'fool', 'tool', and 'school'`
> In the school, the golden rule is to never be a fool. Use your mind as a tool, and always follow the pool.

`Task completed successfully!`

## Example Agent Reply to User - Reference the subtasks' output to answer the user's query
```python
output = my_agent.reply_user()
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
- (Optional) If your description of the variable is one of `int`, `float`, `str`, `dict`, `list`, `array`, `Dict[]`, `List[]`, `Array[]`, `Enum[]`, `bool`, we will enforce type checking when generating the function inputs in `get_next_subtask` method of the `Agent` class. Refer to Tutorial 0 - StrictJSON for the details.
- Inputs (compulsory):
    - **fn_description**: String. Function description to describe process of transforming input variables to output variables. Variables must be enclosed in <> and listed in order of appearance in function input.
    - **output_format**: String. Dictionary containing output variables names and description for each variable. Refer to StrictJSON-Overview.ipynb for details on type checking for `output_format`
    
- Inputs (optional):
    - **examples** - Dict or List[Dict]. Examples in Dictionary form with the input and output variables (list if more than one)
    - **external_fn** - Python Function. If defined, instead of using LLM to process the function, we will run the external function. 
        If there are multiple outputs of this function, we will map it to the keys of `output_format` in a one-to-one fashion
    - **fn_name** - String. If provided, this will be the name of the function. Ohterwise, if `external_fn` is provided, it will be the name of `external_fn`. Otherwise, we will use LLM to generate a function name from the `fn_description`
    - **kwargs** - Dict. Additional arguments you would like to pass on to the `strict_json` function
        
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

#### Example fn_description inferred from type hints and docstring of External Function
```python
# Docstring must provide all compulsory input variables
# We will ignore shared_variables, *args and **kwargs
def add_number_to_list(num1: int, num_list: list, other_var: bool = True, *args, **kwargs):
    '''Adds num1 to num_list'''
    num_list.append(num1)
    return num_list

fn = Function(external_fn = add_number_to_list, 
    output_format = {'num_array': 'Array of numbers'})

str(fn)
```

#### Example Output
`Description: Adds <num1: int> to <num_list: list>`

`Input: ['num1', 'num_list']`

`Output: {'num_list': 'Array of numbers'}`

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

> {'output1': 9}

`Subtask identified: Generate a happy sentence with the decimal number and a ball`
`Calling function sentence_with_objects_entities_emotion with parameters {'obj': '9', 'entity': 'ball', 'emotion': 'happy'}`

> {'output': 'I am so happy with my 9 balls.'}

`Task completed successfully!`

# Inception: Agents within Agents
- You can also create a Meta Agent that uses other Agents (referred to as Inner Agents) as functions
- Create your Meta agent using `Agent()` (Note: No different from usual process of creating Agents - your Meta Agent is also an Agent)
- Set up an Inner Agent list and assign it to your Meta agent using `assign_agents(agent_list)`

## Example Meta Agent Setup
```python
# Define your meta-agent
my_agent = Agent('Menu Creator', 
                 'Creates a menu for a restaurant. Menu item includes Name, Description, Ingredients, Pricing.')

# Define your agent list. Note you can just assign functions to the agent in place using .assign_functions(function_list)
agent_list = [
    Agent('Chef', 'Takes in dish names and comes up with ingredients for each of them. Does not generate prices.'),
    Agent('Boss', 'Takes in menu items and curates them according to price'),
    Agent('Creative Writer', 'Takes in a cuisine type and generates interesting dish names and descriptions. Does not generate prices or ingredients.', max_subtasks = 2),
    Agent('Economist', 'Takes in dish names and comes up with fictitious pricing for each of them')
    ]

my_agent.assign_agents(agent_list)
```

## Run the Meta Agent
- Let us run the agent and see the interactions between the Meta Agent and Inner Agents to solve the task!
```python
output = my_agent.run('Give me 5 menu items with name, description, ingredients and price based on Italian food choices.')
```

# Shared Variables

*"Because text is not enough"* - Anonymous

- `shared_variables` is a dictionary, that is initialised in Agent (default empty dictionary), and can be referenced by any function of the agent (including Inner Agents and their functions)
- This can be useful for non-text modalitiies (e.g. audio, pdfs, image) and lengthy text modalities, which we do not want to output into `subtasks_completed` directly
- `s_` at the start of the variable names means shared variables
    - For input, it means we take the variable from `shared_variables` instead of LLM generated input
    - For output, it means we store the variable into `shared_variables` instead of storing it in `subtasks_completed`. If `subtasks_completed` output is empty, it will be output as `{'Status': 'Completed'}`
- Example shared variables names: `s_sum`, `s_total`, `s_list_of_words`

## Example Input
```python
# Function takes in increment (LLM generated) and s_total (retrieves from shared variable dict), and outputs to s_total (in shared variable dict)
add = Function(fn_description = "Add <increment: int> to <s_total>", 
              output_format = {"s_total": "Modified total"})

# Define the calculator agent and the shared_variables - Note the naming convention of s_ at the start of the names for shared variables
my_agent = Agent('Calculator', 'Does computations', shared_variables = {'s_total': 0}).assign_functions([add])

output = my_agent.run('Increment total by 1')

print('Shared Variables:', my_agent.shared_variables)
```

## Example Output
`Subtask identified: Add 1 to the total`

`Calling function add_int_to_variable with parameters {'increment': 1}`
> {'Status': 'Completed'}

`Task completed successfully!`

`Shared Variables: {'s_total': 1}`

## Example External Function Accessing Shared Variables (Advanced)
```python
# Use shared_variables as input to your external function to access and modify the shared variables
def generate_quotes(shared_variables, number_of_quotes: int, category: str):
    ''' Generates number_of_quotes quotes about category '''
    # Retrieve from shared variables
    my_quote_list = shared_variables['s_quote_list']
    
    ### Add your function code here ###
    
    # Store back to shared variables
    shared_variables['s_quote_list'] = my_quote_list

generate_quote_fn = Function(output_format = {}, external_fn = generate_quotes)
```

# Memory

## Key Philosophy
- It would be important to learn from past experience and improve the agentic framework - memory is key to that
- You can add to the memory bank of your Agents pre-inference (by collecting from a pool of data prior to running the Agent), or during inference (add on in between running subtasks)

## Use Memory in Agents
- Agent class takes `memory_bank` as a parameter during initialisation of an `Agent`
- memory_bank: class Dict[Memory]. Stores multiple types of memory for use by the agent. Customise the Memory config within the Memory class.
    - Default: `memory_bank = {'Function': Memory(top_k = 5, mapper = lambda x: x.fn_description, approach = 'retrieve_by_ranker')}`
    - Key: `Function` (Already Implemented Natively) - Does RAG over Task -> Function mapping
    - Can add in more keys that would fit your use case. Retrieves similar items to task/overall plan (if able) for additional context in `get_next_subtasks()` and `use_llm()` function
    - Side Note: RAG can also be done (and may be preferred) as a separate function of the Agent to retrieve more information when needed (so that we do not overload the Agent with information)

## Memory Class
- Retrieves top k memory items based on task 
- Inputs:
    - `memory`: List. Default: Empty List. The list containing the memory items
    - `top_k`: Int. Default: 3. The number of memory list items to retrieve
    - `mapper`: Function. Maps the memory item to another form for comparison by ranker or LLM. Default: `lambda x: x`
        - Example mapping: `lambda x: x.fn_description` (If x is a Class and the string you want to compare for similarity is the fn_description attribute of that class)
    - `approach`: str. Either `retrieve_by_ranker` or `retrieve_by_llm` to retrieve memory items.
        - Ranker is faster and cheaper as it compares via embeddings, but are inferior to LLM-based methods for context information
    - `ranker`: `Ranker`. The Ranker which defines a similarity score between a query and a key. Default: OpenAI `text-embedding-3-small` model. 
        - Can be replaced with a function which returns similarity score from 0 to 1 when given a query and key
        
## Example Use Case
- Helps to reduce number of functions present in LLM context for more accurate generation
```python
output = my_agent.run('Calculate 2**10 * (5 + 1) / 10')
```

`Original Function List: add_numbers, subtract_numbers, add_three_numbers, multiply_numbers, divide_numbers, power_of, GCD_of_two_numbers, modulo_of_numbers, absolute_difference, generate_poem_with_numbers, List_related_words, generate_quote`

`Filtered Function Names: add_three_numbers, multiply_numbers, divide_numbers, power_of, modulo_of_numbers`

# Known Limitations
1. As the agent uses the term "Overall Plan" for its internal planning, try not to use the word "plan" in your query or context, if not it might confuse the agent. Use alternative words like "schedule"

# Contributing to the project

## Test locally
1. Clone the repository
2. If using a virtual environment, activate it
3. `cd` into taskgen repository
4. Install the package via command line `pip install -e .`
5. Now you can import the package and use it in your code

## Submitting a pull request
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Push your changes to your fork
5. Submit a pull request

# What are we looking out for?
1. Integrations with functions - It would be good if we could import function definitions from elsewhere, e.g. LangChain, into the format shown here. It might even be done automatically using LLM-based conversion using StrictJSON!
2. Jupyter Notebooks showcasing what could be done with the framework for something useful. Let your imagination guide you, we look forward to see what you create
3. Other Known Limitations - Do test the framework out extensively and note its failure cases. We will see if we can address them, if not we will put them in Known Limitations.
4. (For the prompt engineer). If you could find a better way to make the prompts work, let us know directly - we do need to test this out across all Tutorial Jupyter Notebooks to make sure that it really works with existing datasets. Also, if you are using other LLMs beside OpenAI, and find the prompts do not work as well - try to rejig your own prompts and let us know as well!
