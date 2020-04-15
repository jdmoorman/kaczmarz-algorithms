class Cyclic:
    def __init__(self, A, *args, **kwargs):
        self.n_rows = A.shape[0]
        self.row_index = -1

    def next_row_index(self, *args, **kwargs):
        self.row_index = (1 + self.row_index) % self.n_rows
        return self.row_index
