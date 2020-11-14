

from typing import Dict, List, Any

UID = int


class RefCountedList():

    def __init__(self):
        self._dict: Dict[UID, Any] = {}
        self._max_uid: UID = 0
        self._free_uid: List[UID] = []

    def __get_uid(self) -> UID:
        if self._free_uid:
            return self._free_uid.pop()
        uid: UID = self._max_uid
        self._max_uid += 1
        return uid

    def append(self, element: Any, refs: int = 1) -> UID:
        uid: UID = self.__get_uid()
        self._dict[uid] = [element, refs]
        return uid

    def increase_refs(self, uid: UID, new_refs: int = 1):
        self._dict[uid][1] += new_refs

    def decrease_refs(self, uid: UID, removed_refs: int = 1) -> bool:
        tup = self._dict[uid]
        tup[1] -= removed_refs
        if tup[1] <= 0:
            del self._dict[uid]
            self._free_uid.append(uid)
            return True
        return False

    def tolist(self) -> List[Any]:
        return [el for (el, _) in self._dict.values()]

    def __getitem__(self, key: UID):
        return self._dict[key][0]
