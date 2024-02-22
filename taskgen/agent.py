import heapq
import openai
from openai import OpenAI
import numpy as np
import copy
from .base import *

### Helper Functions
def top_k_index(lst, k):
    ''' Given a list lst, find the top k indices corresponding to the top k values '''
    indexed_lst = list(enumerate(lst))
    top_k_values_with_indices = heapq.nlargest(k, indexed_lst, key=lambda x: x[1])
    top_k_indices = [index for index, _ in top_k_values_with_indices]
    return top_k_indices

### Main Classes
class Ranker:
    ''' This defines the ranker which outputs a similarity score given a query and a key'''
    def __init__(self, host = "openai", model = "text-embedding-3-small", database = None):
        '''
        host: Str. The host name for the similarity score service
        model: Str. The name of the model for the host
        database: None / Dict (Key is str for query/key, Value is embedding in List[float])
        Takes in database (dict) of currently generated queries / keys and checks them
        so that you do not need to redo already obtained embeddings
        If you provide a database, the calculated embeddings will be added to this database'''
        self.host = host
        self.model = model
        self.database = database
            
    def __call__(self, query, key) -> float:
        ''' Takes in a query and a key and outputs a similarity score 
        Compulsory:
        query: Str. The query you want to evaluate
        key: Str. The key you want to evaluate'''
     
        database_provided = isinstance(self.database, dict)
        if self.host == "openai":
            client = OpenAI()
            if database_provided and query in self.database:
                query_embedding = self.database[query]
            else:
                query = query.replace("\n", " ")
                query_embedding = client.embeddings.create(input = [query], model=self.model).data[0].embedding
                if database_provided:
                    self.database[query] = query_embedding
                
            if database_provided and key in self.database:
                key_embedding = self.database[query]
            else:
                key = key.replace("\n", " ")
                key_embedding = client.embeddings.create(input = [query], model=self.model).data[0].embedding
                if database_provided:
                    self.database[key] = key_embedding
                
        return np.dot(query_embedding, key_embedding)
        
class Memory:
    ''' Retrieves top k information based on task 
    retriever takes in a query and a key and outputs a similarity score'''
    def __init__(self, memory: list = [], ranker = Ranker()):
        self.memory = memory
        self.ranker = ranker
    def extract(self, task, top_k = 3):
        ''' Performs retrieval of top_k similar memories '''
        memory_score = [self.ranker(memory_chunk, task) for memory_chunk in self.memory]
        top_k_indices = top_k_index(memory_score, top_k)
        return [self.memory[index] for index in top_k_indices]
    def extract_llm(self, task, top_k = 3):
        ''' Performs retrieval via LLMs '''
        res = strict_json(f'You are to output the top {top_k} most similar list items in Memory that meet this description: {task}\nMemory: {self.memory}', '', 
              output_format = {f"top_{top_k}_list": f"Array of top {top_k} most similar list items in Memory, type: list[str]"})
        return res[f'top_{top_k}_list']
    def clear_memory(self):
        ''' Clears all memory '''
        self.memory = []
    
class Agent:
    def __init__(self, agent_name: str = 'Helpful Assistant',
                 agent_description: str = 'A generalist agent meant to help solve problems',
                 max_subtasks: int = 5,
                 memory = Memory(),
                 verbose: bool = True,
                 debug: bool = False,
                 **kwargs):
        ''' 
        Creates an LLM-based agent using description and outputs JSON based on output_format. 
        Agent does not answer in normal free-flow conversation, but outputs only concise task-based answers
        Design Philosophy:
        - Give only enough context needed to solve problem
        - Modularise components for greater accuracy of each component
        
        Inputs:
        - agent_name: String. Name of agent, hinting at what the agent does
        - agent_description: String. Short description of what the agent does
        - max_subtasks: Int. The maximum number of subtasks the agent can have
        - memory: class Memory. Stores the agent's memory for use for retrieval purposes later (to be implemented)
        - verbose: Bool. Default: True. Whether to print out intermediate thought processes
        - debug: Bool. Default: Falsee. Whether to debug StrictJSON messages
        
        Inputs (optional):
        - **kwargs: Dict. Additional arguments you would like to pass on to the strict_json function
        
        '''
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.max_subtasks = max_subtasks
        self.verbose = verbose
        self.memory = memory
        
        self.debug = debug
        
        # reset agent's state
        self.reset()
        
        self.kwargs = kwargs
        
        # start with default of only llm as the function
        self.function_map = {}
        self.assign_functions([Function(fn_name = 'use_llm', 
                                        fn_description = f'Used only when no other function can do the task', 
                                        output_format = {"Output": "Output of LLM"}),
                              Function(fn_name = 'end_task',
                                       fn_description = 'Use only when task is completed',
                                       output_format = {})])
        
    ## Generic Functions ##
    def reset(self):
        self.task = 'No task assigned'
        # This is for meta agent's task for hierarchical structure
        self.overall_task = 'No task assigned'
        self.subtasks_completed = {}
        self.task_completed = False
        
    def query(self, query: str, output_format: dict, provide_function_list: bool = False):
        ''' Queries the agent with a query and outputs in output_format. 
        If you want to provide the agent with the context of functions available to it, set provide_function_list to True (default: False)'''
        
        system_prompt = f"You are an agent named {self.agent_name} with the following description: ```{self.agent_description}```\n"
        if provide_function_list:
            system_prompt += f"You have the following Equipped Functions available:\n```{self.list_functions()}```\n"
        system_prompt += query
        
        res = strict_json(system_prompt = system_prompt,
        user_prompt = '',
        output_format = output_format, 
        verbose = self.debug,
        **self.kwargs)
        return res
    
    def status(self):
        ''' Prints prettily the update of the agent's status. 
        If you would want to reference any agent-specific variable, just do so directly without calling this function '''
        print('Agent Name:', self.agent_name)
        print('Agent Description:', self.agent_description)
        print('Available Functions:', list(self.function_map.keys()))
        print('Task:', self.task)
        if len(self.subtasks_completed) == 0: 
            print("Subtasks Completed: None")
        else:
            print('Subtasks Completed:')
            for key, value in self.subtasks_completed.items():
                print(f"Subtask: {key}\n{value}\n")
        print('Is Task Completed:', self.task_completed)
        
    ## Functions for function calling ##
    def assign_functions(self, function_list: list):
        ''' Assigns a list of functions to be used in function_map '''
        if not isinstance(function_list, list):
            function_list = [function_list]
            
        for function in function_list:
            if not isinstance(function, Function):
                raise Exception('Registered function must be of class Function')

            stored_fn_name = function.__name__
            # if function name is already in use, change name to name + '_1'. E.g. summarise -> summarise_1
            while stored_fn_name in self.function_map:
                stored_fn_name += '_1'

            # add in the function into the function_map
            self.function_map[stored_fn_name] = function
            
        return self
        
    def remove_function(self, function_name: str):
        ''' Removes a function from the agent '''
        if function_name in self.function_map:
            del self.function_map[function_name]
        
    def list_functions(self, as_list: bool = False) -> list:
        ''' Returns the list of functions available to the agent '''
        
        return [f'Name: {name}\n' + str(function) for name, function in self.function_map.items()]
    
    def print_functions(self):
        ''' Prints out the list of functions available to the agent '''
        functions = self.list_functions()
        print('\n'.join(functions))
        
    def select_function(self, task: str = ''):
        ''' Based on the task (without any context), output the next function name and input parameters '''
        _, function_name, function_params = self.get_next_subtask(task = task)
            
        return function_name, function_params
    
    def use_agent(self, agent_name: str, agent_task: str):
        ''' Uses an inner agent to do a task for the meta agent. Task outcomes goes directly to subtasks_completed of meta agent '''
        self.use_function(agent_name, {"instruction": agent_task}, agent_task)
        
    def use_function(self, function_name: str, function_params: dict, subtask: str = '', stateful: bool = True):
        ''' Uses the function. stateful means we store the outcome of the function '''
        if function_name == "use_llm":
            if self.verbose: 
                print(f'Getting LLM to perform the following task: {function_params["instruction"]}')

            res = self.query(query = f'Context:```{self.overall_task}-{self.subtasks_completed}```\nTask: ```{function_params["instruction"]}```\nPerform the task - do not just comment on how you can do the task - do it out fully within your Agent Capabilities', 
                            output_format = {"Task Outcome": "Generate a full response to the Task"},
                            provide_function_list = False)
            
            res = res["Task Outcome"]
            
            if self.verbose: 
                print('>', res)
                print()
            
        elif function_name == "end_task":
            return
        
        else:
            if self.verbose: 
                print(f'Calling function {function_name} with parameters {function_params}')
                
            res = self.function_map[function_name](**function_params)

            # if only one Output key for the json, then omit the output key
            if len(res) == 1 and "Output" in res:
                res = res["Output"]
            
            if self.verbose and res != '': 
                print('>', res)
                print()
                
        if stateful:
            if res != '':
                self.subtasks_completed[subtask] = res

        return res
    
    ## Functions for Task Solving ##
    def assign_task(self, task: str, overall_task: str = ''):
        ''' Assigns a new task to this agent. Also treats this as the meta agent now that we have a task assigned '''
        self.task = task
        self.overall_task = task
        if overall_task != '':
            self.overall_task = overall_task
            
        self.task_completed = False
        
    def get_next_subtask(self, task = '', force: bool = False):
        ''' Based on what the task is and the subtasks completed, we get the next subtask, function and input parameters
        force means we will definitely do something other than end_task'''
        if task == '':
            background_info = f"Assigned Task:```{self.task}```\nAction History: ```{self.subtasks_completed}```\n"
        else:
            background_info = f"Assigned Task:```{task}```"
        
        res = self.query(query = f'''\n{background_info}First create an Overall Plan with a list of steps to do Assigned Task from beginning to end, including uncompleted steps. Then, reflect on Action History to see what steps in Overall Plan are already completed. Then, generate the Next Step to fulfil the Assigned Task. Check that the Next Step can be processed by exactly one Equipped Function. If Overall Plan has been completed, call end_task''',
                         output_format = {"Thoughts": "How to do Assigned Task", "Overall Plan": "List of steps to complete Assigned Task from beginning to end, type: list", "Reflection": "Reflect on progress", "Overall Plan Completed": "List [] of steps in Overall Plan that are completed, type: list", "Next Step": "Describe what to do for next step without mentioning the Equipped Function", "Equipped Function": "What Equipped Function to use for next step", "Equipped Function Input": "Input for the Equipped Function in the form of {'input_parameter': 'input_value'} for each 'input_parameter' in Input, type: dict"},
                          provide_function_list = True)
            
        # default just use llm
        if res['Equipped Function'] not in self.function_map:
            res['Equipped Function'] = 'use_llm'
            
        # if task completed, then end it
        if res['Overall Plan'] == res['Overall Plan Completed'] and len(res['Overall Plan']) > 0:
            res['Equipped Function'] = 'end_task'
            
        # if no plan, do not end yet
        if res['Overall Plan'] == [] and res['Equipped Function'] == 'end_task':
            res['Equipped Function'] = 'use_llm'
        
        # if use_llm or end_task is in the Equipped Function, just make it use_llm or end_task
        if 'use_llm' in res['Equipped Function']: 
            res['Equipped Function'] = 'use_llm'
        if 'end_task' in res['Equipped Function']: 
            res['Equipped Function'] = 'end_task'
        
        # if the next step is already done before, then end the task
        if res["Next Step"] in self.subtasks_completed:
            res["Equipped Function"] = "end_task"
            
        # the first pass we should output something if force is activated
        if force and res['Equipped Function'] == 'end_task':
            res['Equipped Function'] = 'use_llm'
        
        # format the parameters if it is blank or use_ll 
        if res["Equipped Function Input"] == {} or res['Equipped Function'] == 'use_llm':
            res["Equipped Function Input"] = {"instruction": res["Next Step"]}
            
        return res["Next Step"], res["Equipped Function"], res["Equipped Function Input"]
    
    def remove_last_subtask(self):
        ''' Removes last subtask in subtask completed. Useful if you want to retrace a step '''
        if len(self.subtasks_completed) > 0:
            removed_item = self.subtasks_completed.popitem()
        if self.verbose:
            print(f'Removed last subtask from subtasks_completed: {removed_item}')
        
    def summarise_subtasks_completed(self, task: str = ''):
        ''' Summarise the subtasks_completed list according to task '''

        output = self.reply_user(task)
        # Create a new summarised subtasks completed list
        self.subtasks_completed = {f"Summary of {task}": output}
        
    def reply_user(self, query: str = '', stateful: bool = True):
        ''' Generate a reply to the user based on the query / agent task and subtasks completed
        If stateful, also store this interaction into the subtasks_completed'''
        
        my_query = self.task if query == '' else query
            
        res = self.query(query = f'Context: ```{self.subtasks_completed}```\nTask: ```{my_query}```\n', 
                                    output_format = {"Task Outcome": "Generate a response to the Task within your Agent capabilities. Use the Context as ground truth."},
                                    provide_function_list = False)
        
        res = res["Task Outcome"]
        
        if self.verbose:
            print(res)
        
        if stateful:
            self.subtasks_completed[my_query] = res
        
        return res

    def run(self, task: str = '', overall_task: str = '', num_subtasks: int = 0) -> list:
        ''' Attempts to do the task using LLM and available functions
        Loops through and performs either a function call or LLM call up to max_steps number of times
        If overall_task is filled, then we store it to pass to the inner agents for more context
        reset = True means that we reset the agent's metadata
        Returns list of outputs for each substep'''
            
        # Assign the task
        if task != '':
            self.task_completed = False
            # If meta agent's task is here as well, assign it too
            if overall_task != '':
                self.assign_task(task, overall_task)
            else:
                self.assign_task(task)
            
        # check if we need to override num_steps
        if num_subtasks == 0:
            num_subtasks = self.max_subtasks
        
        # if task completed, then exit
        if self.task_completed: 
            if self.verbose:
                print('Task already completed!\n')
                print('Subtasks completed:')
                for key, value in self.subtasks_completed.items():
                    print(f"Subtask: {key}\n{value}\n")
                    
        else:
            # otherwise do the task
            for i in range(num_subtasks):           
                # Determine next subtask, or if task is complete. Always execute if it is the first subtask
                subtask, function_name, function_params = self.get_next_subtask(force = (i==0))
                if function_name == 'end_task':
                    self.task_completed = True
                    if self.verbose:
                        print('Task completed successfully!\n')
                    break
                    
                if self.verbose: print('Subtask identified:', subtask)

                # Execute the function for next step
                res = self.use_function(function_name, function_params, subtask)
                         
            # check if overall task is complete at the last step if num_steps > 1
            if not self.task_completed:
                subtask, function_name, function_params = self.get_next_subtask()
                if function_name == "end_task":
                    self.task_completed = True
                    if self.verbose:
                        print('Task completed successfully!\n')
                                  
        return list(self.subtasks_completed.values())
    
    ## This is for Multi-Agent uses
    def to_function(self, meta_agent):
        ''' Converts the agent to a function so that it can be called by another agent
        The agent will take in an instruction, and output the result after processing'''

        # makes the agent appear as a function that takes in an instruction and outputs the executed instruction
        my_fn = Function(fn_name = self.agent_name,
                             fn_description = f'Agent Description: ```{self.agent_description}```\nExecutes the given <instruction>',
                             output_format = {"Output": "Output of instruction"},
                             external_fn = Agent_External_Function(self, meta_agent))
        
        return my_fn
    
    def assign_agents(self, agent_list: list):
        ''' Assigns a list of Agents to the main agent, passing in the meta agent as well '''
        if not isinstance(agent_list, list):
            agent_list = [agent_list]
        self.assign_functions([agent.to_function(self) for agent in agent_list])
        return self
    
class Agent_External_Function:
    ''' Creates a Function-based version of the agent '''
    def __init__(self, agent: Agent, meta_agent: Agent):
        ''' Retains the instance of the agent as an internal variable '''
        self.agent = agent
        self.meta_agent = meta_agent

    def __call__(self, instruction: str):
        ''' Returns what the agent did overall to fulfil the instruction '''
        # make a deep copy so we do not affect the original agent
        if self.agent.verbose:
            print(f'\n### Start of Inner Agent: {self.agent.agent_name} ###')
        agent_copy = copy.deepcopy(self.agent)
        
        # provide the subtasks completed and debug capabilities to the inner agents too
        agent_copy.debug = self.meta_agent.debug
        agent_copy.subtasks_completed = self.meta_agent.subtasks_completed

        output = agent_copy.run(instruction, self.meta_agent.overall_task)
        if self.agent.verbose:
            print(f'### End of Inner Agent: {self.agent.agent_name} ###')
        
        return ''
                             
