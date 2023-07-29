from datetime import datetime, timedelta

def get_backtest_dates(start_date: datetime, end_date: datetime) -> List[datetime]:
    # Initialize a list to store the backtest dates
    backtest_dates = []

    # Calculate the number of days between the start and end dates
    num_days = (end_date - start_date).days

    # Generate the backtest dates
    for i in range(num_days + 1):
        backtest_dates.append(start_date + timedelta(days=i))

    return backtest_dates

def timeit(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start} seconds")
        return result

    return wrapper
