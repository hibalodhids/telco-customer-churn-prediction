#finding the leap year using python.
leap_year = int (input("Enter any year:"))

if leap_year % 4 == 0:
    print(leap_year, "is a leap year")

else:
    print(leap_year, "is not a leap year")