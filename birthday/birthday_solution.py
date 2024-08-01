from datetime import date

def how_old_are_you(birthdate):

  today = date.today()

  year_diff = today.year - birthdate.year

  month_day_diff = ((today.month, today.day) < (birthdate.month, birthdate.day))

  age = year_diff - month_day_diff

  return age

my_birthday = date(1996, 12, 22)

print(how_old_are_you(my_birthday))