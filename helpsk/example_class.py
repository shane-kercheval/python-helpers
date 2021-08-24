"""Documentation
"""


class ExampleClass:
    """ExampleClass Documentation
    """
    def __init__(self, my_variable=0):
        """init documentation

        Args:
            my_variable:
                docs
        """
        self._my_variable = my_variable

    def my_method(self, value):
        """
        my_method Documentation
        """
        return self._my_variable + value

    @property
    def my_variable(self):
        """Getter method"""
        print("getter of my_variable called")
        return self._my_variable

    @my_variable.setter
    def my_variable(self, value):
        """Setter method"""
        print("setter of my_variable called")
        self._my_variable = value
