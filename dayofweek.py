class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        def is_leap_year(year: int) -> bool:
            return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

        def days_in_month(month: int, year: int) -> int:
            regular_year = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            leap_year = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            return leap_year[month - 1] if is_leap_year(year) else regular_year[month - 1]

        def days_between_years(start_year: int, end_year: int) -> int:
            days = 0
            for year in range(start_year, end_year):
                days += 366 if is_leap_year(year) else 365
            return days

        reference_year = 1970
        reference_day_of_week = 4  # Thursday, January 1st, 1970
        
        # Step 1: Calculate total days from reference year to the current year
        total_days = days_between_years(reference_year, year)
        
        # Step 2: Add the days from the months of the current year
        for m in range(1, month):
            total_days += days_in_month(m, year)
        
        # Step 3: Add the days in the current month
        total_days += day - 1  # Exclude the current day, count from day 1
        
        # Step 4: Calculate the day of the week
        day_of_week_index = (reference_day_of_week + total_days) % 7
        days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        return days_of_week[day_of_week_index]













class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        prev_year = year - 1
        print(prev_year)
        days = prev_year * 365 + prev_year // 4 - prev_year // 100 + prev_year // 400
        print(days)
        days += sum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][:month - 1])
        days += day

        if month > 2 and ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
            days += 1
        print(days)

        return ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][days % 7]