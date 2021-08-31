"""Collection of Exception classes. Used to make debugging easier for external callers of the API and to
ensure that unit tests that use self.raiseException test for the correct exception types.
"""

class HelpskError(Exception):
    """Base-class for all exceptions raised by this package."""


class HelpskParamTypeError(HelpskError):
    """There was an invalid parameter type detected."""


class HelpskParamValueError(HelpskError):
    """There was an invalid parameter type detected."""


class HelpskAssertionError(HelpskError):
    """The expected condition was not True."""
