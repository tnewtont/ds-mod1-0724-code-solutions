import datetime as dt

# Thanksgiving is on November 28, 2024

current_day = dt.datetime.now()

thanksgiving_day = dt.datetime(2024, 11, 28)

days_till_thanksgiving = thanksgiving_day - current_day

print(days_till_thanksgiving.days)