from dataclasses import dataclass


@dataclass(init=True, frozen=True, repr=True, eq=True)
class IndexWithStart:
    start: int = 0
    i: int = 0

    def copy(self):
        return IndexWithStart(start=self.start, i=self.i)


@dataclass(init=True, frozen=False, repr=False, eq=False)
class Indexes:
    # TODO class should eventually be removed in favor of the indices directly on the frame.
    frame: IndexWithStart = None
    tween: IndexWithStart = None

    @staticmethod
    def create():
        return Indexes(IndexWithStart(), IndexWithStart())

    @staticmethod
    def create_from_last(last_indexes, i: int):
        """Creates a new `Indexes` object based on the last one, but updates the tween start index."""
        return Indexes(last_indexes.frame, IndexWithStart(last_indexes.tween.start, i))

    def update_tween_start(self, turbo):
        tween_start = max(self.frame.start, self.frame.i - turbo.cadence)
        self.tween = IndexWithStart(tween_start, self.tween.i)

    def update_tween_index(self, i):
        self.tween = IndexWithStart(self.tween.start, i)

    def update_tween_start_index(self, i):
        self.tween = IndexWithStart(i, self.tween.start)

    def update_frame(self, i: int):
        self.frame = IndexWithStart(self.frame.start, i)

    def is_not_first_frame(self):
        return self.frame.i > 0

    def is_first_frame(self):
        return self.frame.i == 0

    def copy(self):
        return Indexes(
            frame=self.frame.copy() if self.frame else None,
            tween=self.tween.copy() if self.tween else None)
