class LoadModelException(BaseException):
    def __init__(self, arg):
        self.strerror = arg
        self.args = {arg}


class SaveModelException(Exception):
    def __init__(self, arg):
        self.strerror = arg
        self.args = {arg}


class RunTimeException(BaseException):
    def __init__(self, arg):
        self.strerror = arg
        self.args = {arg}
