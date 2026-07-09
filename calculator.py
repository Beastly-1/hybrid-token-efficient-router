import ast
import math
import operator


class Calculator:

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "factorial": math.factorial,
        "abs": abs,
        "round": round,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def calculate(self, expression):

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._evaluate(tree.body)

            return {
                "success": True,
                "result": result
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _evaluate(self, node):

        # Numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed.")

        # Binary operators
        elif isinstance(node, ast.BinOp):

            if type(node.op) not in self.OPERATORS:
                raise ValueError("Operator not allowed.")

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            return self.OPERATORS[type(node.op)](left, right)

        # Unary operators (+,-)
        elif isinstance(node, ast.UnaryOp):

            if type(node.op) not in self.OPERATORS:
                raise ValueError("Unary operator not allowed.")

            operand = self._evaluate(node.operand)

            return self.OPERATORS[type(node.op)](operand)

        # Functions
        elif isinstance(node, ast.Call):

            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid function call.")

            func_name = node.func.id

            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Function '{func_name}' not allowed.")

            args = [self._evaluate(arg) for arg in node.args]

            return self.FUNCTIONS[func_name](*args)

        # Constants
        elif isinstance(node, ast.Name):

            if node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]

            raise ValueError(f"Unknown constant '{node.id}'.")

        raise ValueError("Invalid expression.")
calc = Calculator()

print(calc.calculate("2 + 3 * 4"))

print(calc.calculate("sqrt(81)"))

print(calc.calculate("factorial(5)"))

print(calc.calculate("sin(pi/2)"))

print(calc.calculate("log(e)"))

print(calc.calculate("2 ** 10"))

print(calc.calculate("100 // 3"))

print(calc.calculate("100 % 7"))

print(calc.calculate("round(3.14159,2)"))