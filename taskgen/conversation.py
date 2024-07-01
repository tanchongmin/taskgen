from taskgen.agent import Agent
from taskgen.base import strict_json

from termcolor import colored

class ConversableAgent:
    ''' This class takes an Agent and allows for conversational-based interactions with User / another Agent / Environment
    
Also updates persistent memory with latest information in conversation
Inputs:
agent: The agent we want to interact with: Agent
persistent_memory: dict. What kinds of memory the agent should have that persist over the entire conversation and their descriptions
person: Str. The name of the person you are talking to
num_past_conversation: int. The number of past conversations to use for the agent
verbose: bool. Default: True. Whether to print the Agent's inner states'''
    def __init__(self, agent: Agent, persistent_memory: dict = None, person = 'User', num_past_conversation: int = 5, verbose: bool = True):
        self.agent = agent
        self.persistent_memory = persistent_memory
        self.num_past_conversation = num_past_conversation
        self.person = person
        self.verbose = verbose
        
        ''' Define some external variables for the Agent '''
        # add in the various types of memory
        self.agent.shared_variables['Persistent Memory'] = {}
        # add in the conversation
        self.agent.shared_variables['Conversation'] = ['']
        # add in the summary of conversation
        self.agent.shared_variables['Summary of Conversation'] = ''
    
    ## Reply the person
    def chat(self, cur_msg):
        ''' This does one chat with the person, firstly performing actions then replying the person, while updating the important memory '''
        ## Do actions before replying person only if there are actions other than use_llm and end_task
        my_actions = list(self.agent.function_map.keys()) 
        if 'use_llm' in my_actions: my_actions.remove('use_llm')
        if 'end_task' in my_actions: my_actions.remove('end_task')
        if len(my_actions) > 0:
            self.agent.reset()
            self.agent.run(f'''Summary of Conversation: ```{self.agent.shared_variables['Summary of Conversation']}```
Past Conversation: ```{self.agent.shared_variables['Conversation'][-self.num_past_conversation:]}```
Latest input from {self.person}: ```{cur_msg}```
Use Equipped Functions other than use_llm to help answer the latest input from {self.person}''',
            )

        ## Replies the person
        res = self.agent.query(f'''Summary of Conversation: ```{self.agent.shared_variables['Summary of Conversation']}```
Past Conversation: ```{self.agent.shared_variables['Conversation'][-self.num_past_conversation:]}```
Persistent Memory: ```{self.agent.shared_variables['Persistent Memory']}```
Latest input from {self.person}: ```{cur_msg}```
You are in a conversation with {self.person}. 
Use Past Conversation and Persistent Memory and Subtasks Completed as context when replying. Do not hallucinate actions in Subtasks Completed.
First think through how to reply, before drafting the reply.
Thereafter, update the Summary of Conversation''', 
                          
output_format = {"Thoughts": f"How to reply",
                 f"Reply to {self.person}": f"Reply to ```{cur_msg}```",
                 "Summary of Conversation": "Summarise key points of entire conversation in at most two sentences, building on previous Summary"})
        
        # Update the Summary of Conversation and Append the conversation
        self.agent.shared_variables['Summary of Conversation'] = res['Summary of Conversation']
        self.agent.shared_variables['Conversation'].append(f'{self.person}: {cur_msg}')
        self.agent.shared_variables['Conversation'].append(f'{self.agent.agent_name}: {res[f"Reply to {self.person}"]}')
        
        ## Update Persistent Memory
        if self.persistent_memory is not None and self.persistent_memory != {}:
            persistent_memory = strict_json(f'Update all fields of Persistent Memory. Current value: {self.agent.shared_variables["Persistent Memory"]}',
               f'Additional Conversation\n{self.person}: {cur_msg}\n{self.agent.agent_name}: {res[f"Reply to {self.person}"]}',
               output_format = self.persistent_memory,
               model = self.agent.kwargs.get('model', 'gpt-3.5-turbo'),
               llm = self.agent.llm)
                                                           
            self.agent.shared_variables["Persistent Memory"] = persistent_memory
        
        if self.verbose:
            print(colored(f'Thoughts: {res["Thoughts"]}', 'green', attrs = ['bold']))
            print(colored(f'Persistent Memory: {self.agent.shared_variables["Persistent Memory"]}', 'blue', attrs = ['bold']))
            print(colored(f'Summary of Conversation: {res["Summary of Conversation"]}', 'magenta', attrs = ['bold']))
        
        return res[f'Reply to {self.person}']