from enum import Enum


class Xyz(Enum):
    X = "x"
    Y = "y"
    Z = "z"

    def to_i(self):
        if self in Xyz:
            return {Xyz.X: 0, Xyz.Y: 1, Xyz.Z: 2}[self]
        else:
            raise ValueError(f"Invalid input '{self}'. Must be 'x', 'y', or 'z'.")
