from typing import Callable, Any


class SortedList():

    def __init__(self, key: Callable = lambda x: x):
        self._key = key
        self._list = []

    def __getitem__(self, key):
        return self._list.__getitem__(key)

    def _bissect(self, value, start=None, end=None):
        '''
        key(list[a]) <= value < key(list[b])
        '''
        a = start or 0
        b = end or len(self._list)
        c = (a + b) // 2
        while b - a > 1:
            compared = self._key(self._list[c])
            if compared < value:
                b = c
            elif compared > value:
                a = c
            else:
                return c, c + 1
            c = (a + b) // 2
        return a, b

    def append(self, item: Any):
        value = self._key(item)
        a, b = self._bissect(value)
        self._list.insert(b, item)

    def __iter__(self):
        return self._list.__iter__()

    def __len__(self):
        return self._list.__len__()

    def replace(self, index: int, new_element: Any):
        old_value = self._key(self._list[index])
        new_value = self._key(new_element)
        if new_value < old_value:
            self._list.pop(index)
            a, b = self._bissect(new_value, end=index)
            self._list.insert(b, new_element)
        elif new_value > old_value:
            self._list.pop(index)
            a, b = self._bissect(new_value, start=index)
            self._list.insert(b, new_element)
        else:
            self._list[index] = new_element
