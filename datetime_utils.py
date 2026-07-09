from datetime import datetime, timedelta


class DateTimeUtils:

    def current_date(self):
        return datetime.now().date().isoformat()

    def current_time(self):
        return datetime.now().time().strftime("%H:%M:%S")

    def current_datetime(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def day_of_week(self, date):
        try:
            d = datetime.strptime(date, "%Y-%m-%d")
            return d.strftime("%A")
        except ValueError:
            return None

    def add_days(self, date, days):
        try:
            d = datetime.strptime(date, "%Y-%m-%d")
            return (d + timedelta(days=days)).strftime("%Y-%m-%d")
        except ValueError:
            return None

    def subtract_days(self, date, days):
        try:
            d = datetime.strptime(date, "%Y-%m-%d")
            return (d - timedelta(days=days)).strftime("%Y-%m-%d")
        except ValueError:
            return None

    def days_between(self, date1, date2):
        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            return abs((d2 - d1).days)
        except ValueError:
            return None

    def format_date(self, date, format_string):
        try:
            d = datetime.strptime(date, "%Y-%m-%d")
            return d.strftime(format_string)
        except ValueError:
            return None

    def is_leap_year(self, year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def current_timestamp(self):
        return int(datetime.now().timestamp())
dt = DateTimeUtils()

print(dt.current_date())
print(dt.current_time())
print(dt.current_datetime())

print(dt.day_of_week("2026-07-09"))

print(dt.add_days("2026-07-09", 30))
print(dt.subtract_days("2026-07-09", 10))

print(dt.days_between("2026-07-09", "2026-08-09"))

print(dt.format_date("2026-07-09", "%d/%m/%Y"))

print(dt.is_leap_year(2024))
print(dt.is_leap_year(2025))

print(dt.current_timestamp())