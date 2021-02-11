from queue import PriorityQueue

class UniquePriorityQueue:
    """
    UniquePriorityQueue

    A class for a simple priority queue, with no duplicate values (even if they have different
    priorities).
    """
    def __init__(self):
        self._queue = PriorityQueue()
        self._items = set()

    def not_empty(self):
        """
        Returns True iff the queue is not empty
        """
        return len(self._items) > 0

    def put(self, item):
        """
        Add an item to the queue. If the item is already in the queue, will not be added.

        Parameters
        ----------
        item : tuple
            A tuple (priority, value)
        """
        if item[1] in self._items:
            return
        self._items.add(item[1])
        self._queue.put(item)
        return item

    def get(self):
        """
        Get a value from the queue with minimal priority. If the queue is empty returns None.
        """
        if self.not_empty():
            _, item = self._queue.get()
            self._items.remove(item)
            return item
        return None
