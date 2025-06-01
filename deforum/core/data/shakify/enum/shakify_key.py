from enum import Enum


class ShakifyKey(Enum):
    LOC = ("location", "translation")
    ROT = ("rotation_euler", "rotation_3d")

    def shakify(self):
        return self.value[0]

    def deforum(self):
        return self.value[1]
