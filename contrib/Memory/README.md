This enables contributes for various types of memory for the Agent

All kinds of memory must implement the MemoryTemplate base class. See memory.py for example.

Contribute in the form of a Jupyter Notebook implementing the Memory class. If it is good and sufficiently used by others, it will be ported over to the main TaskGen repo with acknowledgement of you as the author :)

```python
class MemoryTemplate(ABC):
    """A generic template provided for all memories"""

    @abstractmethod
    def append(self, memory_list, mapper=None):
        """Appends multiple new memories"""
        pass

    # TODO Should this be deleted based on metadata key - value filter
    @abstractmethod
    def remove(self, existing_memory):
        """Removes an existing_memory. existing_memory can be str, or triplet if it is a Knowledge Graph"""
        pass

    @abstractmethod
    def reset(self):
        """Clears all memories"""

    @abstractmethod
    def retrieve(self, task: str):
        """Retrieves some memories according to task"""
        pass
```
