from tools.calculator import Calculator
from tools.datetime_utils import DateTimeUtils
from tools.json_validator import JSONValidator
from tools.regex_verifier import RegexVerifier
from tools.python_executor import PythonExecutor
from tools.verification import Verification

class ToolRegistry:

    def __init__(self):

        self.tools = {
            "calculator": Calculator(),
            "datetime": DateTimeUtils(),
            "json_validator": JSONValidator(),
            "regex_verifier": RegexVerifier(),
            "python_executor": PythonExecutor(),
            "verification": Verification(),
        }

    def get_tool(self, name):
        return self.tools.get(name)

    def register_tool(self, name, tool):
        self.tools[name] = tool

    def remove_tool(self, name):
        if name in self.tools:
            del self.tools[name]

    def has_tool(self, name):
        return name in self.tools

    def list_tools(self):
        return list(self.tools.keys())
