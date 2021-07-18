class ExampleClass:

    def __init__(self, my_variable=0):
        self._my_variable = my_variable

    @property
    def my_variable(self):
        """I'm the 'my_variable' property."""
        print("getter of my_variable called")
        return self._my_variable

    @my_variable.setter
    def my_variable(self, value):
        print("setter of my_variable called")
        self._my_variable = value
