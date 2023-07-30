HIGH LEVEL OVERVIEW
* 		Setting Up the Environment: Before starting, ensure you have Python installed on your machine. You'll also need to install a few Python libraries that our script depends on. You can install these using pip, Python's package installer. The libraries you'll need are BeautifulSoup, requests, and pandas. You can install them with the following command:Copy codepip install beautifulsoup4 requests pandas 
* 		Creating the Python Files: We'll be creating four Python files, each responsible for a specific task:
    * data_fetcher.py: Responsible for fetching and parsing data from the web.
    * projector.py: Responsible for calculating and projecting game results.
    * backtester.py: Responsible for backtesting and evaluating the projections.
    * result_formatter.py: Responsible for formatting and outputting the results.
* 		Writing the Functions: Each file will contain one or more functions. Here's a brief overview of each function:
    * fetch_and_parse_data(url: str) -> Dict[str, Any]: This function, located in data_fetcher.py, fetches the HTML content of a webpage, handles potential exceptions, optimizes data retrieval by utilizing caching mechanisms, and extracts the relevant data from the HTML content. It uses libraries like requests to fetch the webpage and BeautifulSoup to parse the HTML content. It returns the extracted data in a structured format, such as a dictionary.
    * calculate_and_project(game_days: List[str]) -> Dict[str, str]: This function, located in projector.py, takes as input the list of game days. For each game day, it extracts the home and away teams from the matchups, visits each team's stats page to extract their respective category ranks, compares the ranks, and awards points based on the defined scoring criteria. It then tallies the points for each team to find which team has the most points, and returns a dictionary of matchups and their projected winners.
    * backtest_and_evaluate(backtest_date: str, backtest_period: int, output_filename: str) -> float: This function, located in backtester.py, prompts the user for a backtest period, performs the backtesting process by iterating over the backtest period, fetching the page for each date, extracting the matchup teams, fetching and extracting the team data, calculating the result, and appending the result data to the results list. It then compares the projected winners to the actual results, calculates the win rate, and saves the results to a CSV file. It returns the win rate.
    * format_and_output_results(results: dict, filename: str) -> None: This function, located in result_formatter.py, takes in the results dictionary, modifies the date format, column headers, removes unnecessary columns, adds a new column for point difference, and saves the modified results to a CSV file. It ensures the results are presented in a user-friendly and readable format.
* 		Constants and Dictionaries: In your script, you'll need to define several constants and dictionaries. Constants like MAX_RETRIES, CACHE_EXPIRATION, and API_ENDPOINT are used to control the behavior of the script. Dictionaries like data_dict, cache_dict, and error_dict are used to store data, cached data, and error messages respectively.
* 		Logging and Caching: Implement logging functions (log_error, log_info, log_debug) to keep track of any errors, informational messages, or debug messages. Implement caching functions (cache_data, retrieve_data) to store and retrieve data, which can help optimize the performance of your script.
* 		Running the Script: Once all the functions are implemented, you can run the script. The script will prompt you for a backtest period (default is 7 days), fetch and parse the game data, calculate and project the game results, backtest and evaluate the projections, and finally format and output the results.
Remember, this is a high-level overview. Each function will have its own internal logic and error handling that you'll need to implement. Also, don't forget to test your functions individually to ensure they're working as expected before integrating them into the final script. Good luck!


low-level overview of how to recreate the MLB backtesting script from scratch:

1. **Environment Setup**: Install Python on your machine and ensure you have the necessary libraries: BeautifulSoup, requests, and pandas.

2. **Create Python Files**: Create four Python files: `data_fetcher.py`, `projector.py`, `backtester.py`, and `result_formatter.py`.

3. **Function Signatures**: Define the following functions in each file with their input parameters and return types:
   - `fetch_and_parse_data(url: str) -> Dict[str, Any]` (in `data_fetcher.py`): Fetches webpage content, handles exceptions, optimizes caching, and extracts data.
   - `calculate_and_project(game_days: List[str]) -> Dict[str, str]` (in `projector.py`): Projects game results based on team ranks and scoring criteria.
   - `backtest_and_evaluate(backtest_date: str, backtest_period: int, output_filename: str) -> float` (in `backtester.py`): Performs backtesting and evaluates projections, returning the win rate.
   - `format_and_output_results(results: dict, filename: str) -> None` (in `result_formatter.py`): Formats and saves results to a CSV file.

4. **Logic Inside Each Function**: Describe the logic in each function:
   - `fetch_and_parse_data`: Fetches the HTML content, parses relevant data, and returns it as a dictionary.
   - `calculate_and_project`: Processes game days, extracts teams, visits stats pages, compares ranks, and determines winners.
   - `backtest_and_evaluate`: Performs backtesting by fetching data and comparing projections to actual results.
   - `format_and_output_results`: Formats the data and saves it to a CSV file.

5. **Examples of Using Functions**: Provide usage examples for each function.

6. **Functions Interaction**: Describe how functions should be used together in the correct order.

7. **Data Details**: Explain the format, type, and constraints of data stored in dictionaries.

8. **Constants**: Define constants like `MAX_RETRIES`, `CACHE_EXPIRATION`, and `API_ENDPOINT`.

9. **Caching and Logging**: Explain caching mechanism for data storage and logging for error, info, and debug messages.

10. **Running the Script**: Summarize the steps to run the script, which prompts for a backtest period, performs calculations, and outputs results.

11. **Testing**: Emphasize the importance of testing functions individually before integrating them.

A low-level overview provides more granular details, helping beginners understand the implementation steps and the functionalities of each component in the script.


GENERAL OVERVIEW:

the scoring is based on the difference in ranks between two teams in each category, and the lower-ranking team gets a point for each category where they have a better rank than the higher-ranking team. The scoring_criteria dictionary in the code should not be converted to float values since it is simply a dictionary with the category names and their associated points. the function will count the points for the lower-ranking team in each category, where the rank is represented by a positive integer. If the rank is not available or the value is negative, it will be skipped. ; the updated calculate_score function contains an error. Both rank_1 and rank_2 are assigned the same value stats[stat_name], which means they represent the rank of the same team in the same category.
To fix this issue, we need to modify the function to retrieve the ranks of both teams (home team and away team) for each category and then calculate the score based on their ranks.
In this corrected version, the function now takes two dictionaries as input: home_team_stats and away_team_stats, representing the statistics and ranks of the home team and away team, respectively. It loops through the scoring_criteria dictionary and checks if the given stat_name exists in both home_team_stats and away_team_stats. If the category is available for both teams, it retrieves their respective ranks as rank_1 (home team) and rank_2 (away team).
The score is then updated based on the difference between the ranks of the two teams in each category. If the away team has a better rank (i.e., a lower rank) than the home team, the weight is added to the score.
With this correction, the function now calculates the score correctly based on the difference in ranks between the two teams in each category, following your scoring criteria.

*

Optimize data retrieval by utilizing caching mechanisms to reduce unnecessary requests and improve performance:
   - Fetch the HTML content of the web page by making an HTTP request using libraries like `aiohttp`.
   - Handle potential exceptions, such as network errors or timeouts, and retry failed requests if necessary.
   - Return the specific HTML content.
   - Extract the relevant data from the HTML content of a web page using libraries like BeautifulSoup.
   - Define appropriate parsing techniques, such as CSS selectors or XPath, to locate relevant elements in the HTML.
   - Extract the required data and return it in a structured format, such as a dictionary or list.
   - Fetch and extract the team data for a given team name by combining the above functions.
   - Construct the URL based on the team name and pass it to the `fetch_page` function to retrieve the HTML content.
   - Pass the HTML content to the `extract_data` function to extract the relevant team data.

Take the extracted team data as input and apply the defined scoring criteria and stat categories to calculate the projection score:
   - Iterate over the categories and scoring criteria, retrieve the corresponding values from the team data, and calculate the weighted score for each category.
   - Aggregate the weighted scores to obtain the final projection score for the team and return it.

* the scoring is based on the difference in ranks between two teams in each category, and the lower-ranking team gets a point for each category where they have a better rank than the higher-ranking team. The scoring_criteria dictionary in the code should not be converted to float values since it is simply a dictionary with the category names and their associated points. the function will count the points for the lower-ranking team in each category, where the rank is represented by a positive integer. If the rank is not available or the value is negative, it will be skipped. ; the updated calculate_score function contains an error. Both rank_1 and rank_2 are assigned the same value stats[stat_name], which means they represent the rank of the same team in the same category.
* To fix this issue, we need to modify the function to retrieve the ranks of both teams (home team and away team) for each category and then calculate the score based on their ranks.
* In this corrected version, the function now takes two dictionaries as input: home_team_stats and away_team_stats, representing the statistics and ranks of the home team and away team, respectively. It loops through the scoring_criteria dictionary and checks if the given stat_name exists in both home_team_stats and away_team_stats. If the category is available for both teams, it retrieves their respective ranks as rank_1 (home team) and rank_2 (away team).
* The score is then updated based on the difference between the ranks of the two teams in each category. If the away team has a better rank (i.e., a lower rank) than the home team, the weight is added to the score.
With this correction, the function now calculates the score correctly based on the difference in ranks between the two teams in each category, following your scoring criteria.
Save the backtest results or other relevant data to a file in a structured format, such as CSV, using the `csv` module:
   - Open the file for writing, iterate over the results, and write them to the file.

Make the following changes to the results dictionary:
   - Change the date format to MM/DD: Modify the line `result["date"] = current_date.strftime('%Y-%m-%d')` to `result["date"] = current_date.strftime('%m/%d')`.
   - Change 'home_team' column header to 'Home': Modify the line `"home_team": home_team` to `"Home": home_team` when constructing the results dictionary.
   - Change 'away_team' column header to 'Away': Modify the line `"away_team": away_team` to `"Away": away_team` when constructing the results dictionary.
   - Remove 'result' column: Remove the line `"result": result` when constructing the results dictionary.
   - Remove 'matchup_count' column: Remove the line `"matchup_count": len(matchups)` when constructing the results dictionary.
   - Change 'Projected Winner' column header to 'Projection': Modify the line `"Projected Winner": result["home_team"] if result["result"] > 0 else result["away_team"]` to `"Projection": result["home_team"] if result["result"] > 0 else result["away_team"]` when constructing the results dictionary.
   - Add 'Point Diff.' column: After calculating the result for each matchup, calculate the point difference between the home and away teams based on your scoring criteria. Add a new key-value pair to the results dictionary with the key `"Point Diff."` and the value representing the point difference.

Column Header and Result Modifications: Change the date format to MM/DD. Change 'home_team' column header to 'Home'. Change 'away_team' column header to 'Away'. Remove 'result' and 'matchup_count' columns. Change 'Projected Winner' column header to 'Projection'. Add 'Point Diff.' column and calculate the point difference between home and away teams.
Perform the backtesting process based on the provided backtest date, backtest period, and output filename:
   - Iterate over the backtest period, fetching the page for each date, extracting the matchup teams, fetching and extracting the team data, calculating the result, and appending the result data to the `results` list.
   - Save the results to a CSV file.
   - Compare the actual winners with the projected winners to calculate the win rate or other performance metrics.
   - Iterate over the matchups or games, compare the actual winners with the projected winners, and calculate the win rate or other metrics based on the number of correct predictions.

Efficiency:
* Concurrency Optimization, Logging, and Code Organization: Implement concurrency optimization techniques, such as using semaphores, to limit concurrent requests and improve performance. Identify critical sections of the code that can benefit from parallelization and distribute computations effectively. Replace print statements with proper logging to improve error reporting and debugging capabilities. Implement comprehensive error handling, including try-except blocks around network requests, to gracefully handle exceptions. Utilize error tracking services like Sentry to receive alerts and track application issues proactively. Split the script into multiple modules or files to achieve better code organization and maintainability. Create separate modules for web scraping, data processing, logging, and database operations to promote modularity. Refactor code to eliminate code duplication and consolidate similar functionality for improved efficiency.
* Data Handling and Optimization: Store data that changes throughout the pipeline in a dictionary, and data that remains constant after scraping in a tuple. Implement concurrency optimization techniques using semaphores to limit concurrent requests and improve performance. Use aiohttp's ClientSession to make concurrent requests, speeding up the overall process. Use a ThreadPoolExecutor to run the extract_matchup_teams() and fetch_and_extract_team_statistics() functions in parallel, further improving performance. Implement a caching mechanism to store data from the fetch_matchups() function to avoid redundant requests and improve performance.
* Caching and Data Parsing: Define the CachingClientSession class to add caching functionality for HTTP requests. Use caching to store and retrieve responses, checking the cache before making requests. Implement cache invalidation or expiration mechanisms to keep data up-to-date. Use appropriate caching strategies, such as in-memory caching or persistent caching with libraries like Redis or Memcached. Ensure that data is properly parsed using libraries like BeautifulSoup and defining appropriate parsing techniques.
Reliability:
* Proper error handling: Use proper error handling to catch and handle exceptions. This will help to ensure that the code does not crash or produce unexpected results. For example, use try-except blocks to catch exceptions that are thrown by the code. Use logging to log any exceptions that are not caught.

* Functionality and Error Handling: Ensure that all function type hinting is correct throughout the script. Properly define and call all functions. Remove or alter any functions that are not being called. Change calls to functions that are not defined to similar existing functions. Ensure that all positional arguments are consistent throughout the script's flow for correct data passing. Add error handling and retry logic to handle network issues gracefully, including catching exceptions during network requests and implementing retry mechanisms.
Scalability:
* Efficient algorithms and data structures: Use efficient algorithms and data structures to ensure that the code can handle increasing amounts of data. For example, use a caching mechanism to store data that is frequently accessed. Use a data structure that is appropriate for the type of data that is being stored.
* Modular design: Use a modular design that makes it easy to add new features. This will help to ensure that the code can be scaled up as needed. For example, create separate modules for web scraping, data processing, logging, and database operations.
* Version control system: Use a version control system, such as Git, to track changes to the code. This will help to identify and revert changes that make the code less scalable.
Maintainability:
* Clear and concise code: Use clear and concise code to make the code easy to understand and modify. This will help to ensure that the code can be maintained by other developers. For example, use descriptive variable names and comments to explain the purpose of the code.
* Consistent coding standards: Use consistent coding standards to make the code easy to read and understand. This will help to ensure that the code is consistent and easy to maintain. For example, use a consistent style for indentation and whitespace.
* Good documentation: Provide good documentation to explain the purpose of the code and how it works. This will help to ensure that the code can be understood and maintained by other developers. For example, write docstrings for all functions and classes to explain their purpose and usage.
* Code Formatting and Documentation: Review the script for PEP 8 compliance and adjust code formatting as necessary. Add type annotations to function signatures for better code understanding and enable static type checking. Include detailed comments and docstrings to explain the purpose, parameters, and return values of functions for better code comprehension.