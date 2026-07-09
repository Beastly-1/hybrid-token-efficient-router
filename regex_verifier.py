import re


class RegexVerifier:

    EMAIL = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    PHONE = r"^\+?[1-9]\d{9,14}$"

    URL = (
        r"^https?://"
        r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
        r"(?:/[^\s]*)?$"
    )

    IPV4 = (
        r"^((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
    )

    UUID = (
        r"^[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}$"
    )

    DATE = r"^\d{4}-\d{2}-\d{2}$"

    TIME = r"^\d{2}:\d{2}(:\d{2})?$"

    def verify(self, pattern, text):
        return re.fullmatch(pattern, text) is not None

    def contains(self, pattern, text):
        return re.search(pattern, text) is not None

    def extract(self, pattern, text):
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def extract_all(self, pattern, text):
        return re.findall(pattern, text)

    def replace(self, pattern, replacement, text):
        return re.sub(pattern, replacement, text)

    def compile(self, pattern):
        return re.compile(pattern)

    def is_email(self, text):
        return self.verify(self.EMAIL, text)

    def is_phone(self, text):
        return self.verify(self.PHONE, text)

    def is_url(self, text):
        return self.verify(self.URL, text)

    def is_ipv4(self, text):
        return self.verify(self.IPV4, text)

    def is_uuid(self, text):
        return self.verify(self.UUID, text)

    def is_date(self, text):
        return self.verify(self.DATE, text)

    def is_time(self, text):
        return self.verify(self.TIME, text)