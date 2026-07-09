import json
from jsonschema import validate, ValidationError

class JSONValidator:

    def validate(self, text):
            try:
                json.loads(text)
                return True
            except json.JSONDecodeError:
                return False

    def parse(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def validate_schema(self, data, schema):
        try:
            validate(instance=data, schema=schema)
            return True
        except ValidationError:
            return False

    def pretty(self, data):
        return json.dumps(data, indent=4)

validator = JSONValidator()

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    },
    "required": ["name", "age"]
}

data1 = {
    "name": "Harshith",
    "age": 19
}

data2 = {
    "name": "Harshith"
}

print(validator.validate_schema(data1, schema))
print(validator.validate_schema(data2, schema))