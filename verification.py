from tools.json_validator import JSONValidator
from tools.regex_verifier import RegexVerifier
from tools.calculator import Calculator
from tools.datetime_utils import DateTimeUtils


class Verification:

    def __init__(self):

        self.json_validator = JSONValidator()
        self.regex_verifier = RegexVerifier()
        self.calculator = Calculator()
        self.datetime = DateTimeUtils()

    # -------------------------
    # JSON
    # -------------------------

    def verify_json(self, text):
        return self.json_validator.validate(text)

    def verify_json_schema(self, data, schema):
        return self.json_validator.validate_schema(data, schema)

    # -------------------------
    # Regex
    # -------------------------

    def verify_regex(self, pattern, text):
        return self.regex_verifier.verify(pattern, text)

    def verify_email(self, text):
        return self.regex_verifier.is_email(text)

    def verify_phone(self, text):
        return self.regex_verifier.is_phone(text)

    def verify_url(self, text):
        return self.regex_verifier.is_url(text)

    def verify_ipv4(self, text):
        return self.regex_verifier.is_ipv4(text)

    def verify_uuid(self, text):
        return self.regex_verifier.is_uuid(text)

    def verify_date(self, text):
        return self.regex_verifier.is_date(text)

    def verify_time(self, text):
        return self.regex_verifier.is_time(text)

    # -------------------------
    # Calculator
    # -------------------------

    def verify_math(self, expression, expected):

        result = self.calculator.calculate(expression)

        if not result["success"]:
            return False

        return result["result"] == expected

    # -------------------------
    # Date / Time
    # -------------------------

    def verify_day(self, date, expected):
        return self.datetime.day_of_week(date) == expected

    def verify_days_between(self, d1, d2, expected):
        return self.datetime.days_between(d1, d2) == expected

    # -------------------------
    # Generic
    # -------------------------

    def verify_equal(self, actual, expected):
        return actual == expected

    def verify_contains(self, text, substring):
        return substring in text

    def verify_not_empty(self, value):
        return value is not None and value != ""