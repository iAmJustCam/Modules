import asyncio
from datetime import datetime
from data_fetching import get_all_matchups
from data_extraction import actual_winner
from data_processing import process_matchups, run_backtester, calculate_score, calculate_win_rate, group_matchups_by_date
from data_writing import write_to_csv
from utility import get_backtest_dates, timeit

@timeit
def run_script(start_date: datetime, end_date: datetime) -> None:
    # Get the backtest dates
    backtest_dates = get_backtest_dates(start_date, end_date)

    # Initialize a list to store all the matchups
    all_matchups = []

    # Fetch and process the matchups for each backtest date
    for date in backtest_dates:
        # Fetch the matchups for the date
        matchups = asyncio.run(get_all_matchups(date))

        # Process the matchups
        processed_matchups = process_matchups(matchups)

        # Add the processed matchups to the list of all matchups
        all_matchups.extend(processed_matchups)

    # Group the matchups by date
    matchups_by_date = group_matchups_by_date(all_matchups)

    # Run the backtester
    backtester_results = run_backtester(matchups_by_date)

    # Calculate the win rate
    win_rate = calculate_win_rate(backtester_results)

    # Write the backtester results to a CSV file
    write_to_csv('backtester_results.csv', backtester_results, ['date', 'home', 'away', 'score', 'actual_winner'])

    # Print the win rate
    print(f"Win rate: {win_rate * 100}%")
