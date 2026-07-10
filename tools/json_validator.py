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

