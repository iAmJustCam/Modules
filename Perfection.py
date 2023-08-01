import asyncio
from configparser import RawConfigParser
from datetime import datetime, timedelta
import aiohttp
from aiohttp import ClientSession, ClientError as AiohttpClientError, ClientTimeout
from bs4 import BeautifulSoup
import urllib
from urllib.parse import quote_plus
from cachetools import cached, TTLCache
from dateutil.relativedelta import relativedelta
import argparse
from typing import List, Dict, Any, TypeVar, Optional
import traceback
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from slugify import slugify
from fuzzywuzzy import process
import random
import logging
import re 
import csv
import os


# Define constants
RANKING_MAX = 31  
DIV = 'div'
DATA_CONTAINER = 'data-container'
TEAM_BASE_URL = os.getenv("TEAM_BASE_URL", "https://www.teamrankings.com/mlb/team/")
VALID_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

# Define data keys
TEAM_DATA_KEY = "team_data"

# MatchupProcessor class
T = TypeVar('T')

# Define custom exceptions
class CustomClientError(Exception):
    """Raised when there is an error with the client."""
    pass

class NoMatchupsError(Exception):
    """Raised when there are no matchups."""
    pass

class NoDataError(Exception):
    """Raised when there is no data."""
    pass

# Define data classes
@dataclass
class Team:
    name: str

@dataclass
class Matchup:
    home: Team
    away: Team
    matchup: str
    date: datetime

    def __init__(self, home: Team, away: Team, matchup: str, date: datetime):
        if not date:
            raise ValueError("Date is required")
        self.date = date # Require date
        self.home = home
        self.away = away
        self.matchup = matchup

@dataclass
class Projection:
    date: datetime
    home: Team
    away: Team
    winner: Team

Results = Dict[str, Any]
Projection = Dict[str, str]


@dataclass
class Config:
    team_name_mapping: Dict[str, str]
    stats_categories: List[str]
    rank_difference: Dict[str, int]
    log_level: str

class ConfigBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **kwargs) -> Config:
        self._instance = Config(**kwargs)
        return self._instance

class ConfigParser:
    @staticmethod
    def parse_config_file(filepath: str) -> Dict[str, str]:
        parser = RawConfigParser()
        parser.read(filepath)

        required_sections = ["team_name_mapping", "stats_categories", "rank_difference", "logging"]
        for section in required_sections:
            if not parser.has_section(section):
                raise Exception(f"Section '{section}' not found in config file")

        team_name_mapping = {k.strip().lower(): v.strip() for k, v in parser.items('team_name_mapping')}
        stats_categories = parser.get("stats_categories", "categories").split(',')
        rank_difference = {k: int(v) for k, v in parser.items("rank_difference")}
        log_level = parser.get("logging", "level")

        return {"team_name_mapping": team_name_mapping, "stats_categories": stats_categories, 
                "rank_difference": rank_difference, "log_level": log_level}

class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._default_config_path()

    def load(self) -> Config:
        config_dict = ConfigParser.parse_config_file(self.config_path)
        config = ConfigBuilder()(**config_dict)
        self._validate_config(config)
        return config

    @staticmethod
    def _default_config_path() -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, 'config.ini')

    @staticmethod
    def _validate_config(config: Config):
        if config.log_level.upper() not in VALID_LOG_LEVELS:
            raise ValueError(f"Invalid log level: {config.log_level}")
        # Add more validation logic here if needed



@dataclass
class BacktestArgs:
    days: int
    config_path: Optional[str]

class ArgsParser:
    @staticmethod
    def parse_args() -> BacktestArgs:
        parser = argparse.ArgumentParser(description="Run the backtest.")
        parser.add_argument('days', type=int, nargs='?', default=7, help='The backtest period in days. Default is 7.')
        parser.add_argument('--config', type=str, help='Path to the config file.')
        args = parser.parse_args()
        return BacktestArgs(args.days, args.config)

class DateCalculator:
    @staticmethod
    def calculate_start_date(days: int) -> datetime:
        return datetime.now() - timedelta(days=days + 1)

    @staticmethod
    def calculate_end_date() -> datetime:
        return datetime.now() - timedelta(days=1)

class Backtest:
    def __init__(self, days: int, config_path: Optional[str]):
        self.start_date = DateCalculator.calculate_start_date(days)
        self.end_date = DateCalculator.calculate_end_date()
        self.config = ConfigLoader(config_path).load()



class StringNormalizer:
    ABBREVIATIONS = {
        'stl': 'st louis',
        'sf': 'san francisco',
        # Add more abbreviations as needed
    }

    CHOICES = list(config.team_name_mapping.keys())  # Use the config instance
    SCORE_CUTOFF = 80
    MATCH_CACHE = OrderedDict()
    
    @classmethod
    def normalize(cls, string: str) -> str:
        # Remove whitespace and convert to lowercase
        normalized_string = string.strip().lower()

        # Remove non-alphanumeric characters
        normalized_string = re.sub(r'\W+', '', normalized_string)

        # Map common abbreviations
        if normalized_string in cls.ABBREVIATIONS:
            normalized_string = cls.ABBREVIATIONS[normalized_string]

        return normalized_string

    @classmethod
    def fuzzy_match(cls, string: str) -> str:
        # Check if we've already matched this string
        if string in cls.MATCH_CACHE:
            return cls.MATCH_CACHE[string]

        # Use FuzzyWuzzy's process.extractOne() to find the best match
        best_match = process.extractOne(string, cls.CHOICES, score_cutoff=cls.SCORE_CUTOFF)
        if best_match:
            logging.info(f"Fuzzy matched string: {best_match[0]} with a score of {best_match[1]}")
            cls.MATCH_CACHE[string] = best_match[0]

            # Limit cache size to 1000 entries
            if len(cls.MATCH_CACHE) > 1000:
                cls.MATCH_CACHE.popitem(last=False)

            return best_match[0]
        else:
            logging.info(f"No fuzzy match found for string: {string}. Using original string.")
            cls.MATCH_CACHE[string] = string
            return string


class Fetcher:
    def __init__(self, config, base_url: str = 'https://www.teamrankings.com'):
        self.config = config
        self.team_name_mapping = config.team_name_mapping
        self.base_url = base_url
        self.team_stats_cache = TTLCache(maxsize=30, ttl=3600)  # cache expires after 1 hour

    def get_team_url(self, team_name: str) -> str:
        # Use fuzzy matching to find the closest matching team name
        match, score = process.extractOne(team_name, self.config.team_name_mapping.keys())
        logging.info(f'Original team name: {team_name}')
        logging.info(f'Fuzzy matched string: {match} with a score of {score}')
        
        # Use the matched team name to get the correct URL slug
        url_slug = self.team_name_mapping[match]
        
        # Generate the URL
        url = f'{self.base_url}/mlb/team/{url_slug}/stats'
        logging.info(f'Generated URL for {team_name}: {url}')
        return str(url)  # Ensure the URL is a string

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    async def fetch_team_page(self, session: ClientSession, url: str, timeout: int = 10) -> str:
        async with session.get(url) as response:
            return await response.text()


# Async parsing pages
class Parser:
    def __init__(self, categories: List[str]):
        self.categories = categories

    @cached(cache=TTLCache(maxsize=100, ttl=3600)) 
    def parse_team_data(self, html: str) -> Dict[str, str]:
        soup = BeautifulSoup(html, 'html.parser')
        return self._parse_data(soup, self.categories)

    def _parse_data(self, soup: BeautifulSoup, categories: List[str]) -> Dict[str, str]:
        data = {}
        for category in categories:
            element = soup.find(text=category)  # Find the element containing the category text
            if element:
                # Get the parent element (which should contain the stat value)
                parent_element = element.find_parent('td', class_='stat')
                if parent_element:
                    data[category] = parent_element.get_text().strip()

        return data


class FetcherInterface(Protocol):
    async def fetch_team_page(self, session: ClientSession, url: str) -> str:
        ...

    def get_team_url(self, team_name: str) -> str:
        ...

class ParserInterface(Protocol):
    def parse_team_data(self, data: str) -> Dict[str, T]:
        ...

TeamData = Dict[str, T]

class MatchupProcessor:
    def __init__(self, fetcher: FetcherInterface, parser: ParserInterface, logger: logging.Logger, config: Config):
        self.fetcher = fetcher
        self.parser = parser
        self.logger = logger
        self.config = config

    async def process_matchup(self, session: ClientSession, matchup: Matchup) -> Dict[str, Any]:
        # Check if the matchup is an instance of Matchup
        if not isinstance(matchup, Matchup):
            raise ValueError("matchup must be a Matchup object")

        # Check if session is an instance of ClientSession
        if not isinstance(session, ClientSession):
            raise ValueError("session must be a ClientSession object")

        start_time = time.time()
        self.logger.info(f"Processing matchup: {matchup}")
        try:
            home_data = await self._fetch_and_parse_team_data(session, matchup.home.name)
            away_data = await self._fetch_and_parse_team_data(session, matchup.away.name)
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error while processing matchup: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing matchup: {e}")
            raise
        else:
            processing_time = time.time() - start_time
            self.logger.info(f"Processing time: {processing_time} seconds")
            return {
                'home': matchup.home,
                'home_ranks': home_data,
                'away': matchup.away,
                'away_ranks': away_data,
                'date': matchup.date
            }

    async def _fetch_and_parse_team_data(self, session: ClientSession, team_name: str) -> TeamData:
        team_page = await self.fetcher.fetch_team_page(session, self.fetcher.get_team_url(team_name))
        team_data = self.parser.parse_team_data(team_page)
        if not team_data:
            self.logger.warning(f"No data found for team: {team_name}")
        return team_data

class MatchupProcessorBuilder:
    def __init__(self):
        self.fetcher = None
        self.parser = None
        self.logger = None
        self.config = None

    def set_fetcher(self, fetcher: FetcherInterface) -> 'MatchupProcessorBuilder':
        self.fetcher = fetcher
        return self

    def set_parser(self, parser: ParserInterface) -> 'MatchupProcessorBuilder':
        self.parser = parser
        return self

    def set_logger(self, logger: logging.Logger) -> 'MatchupProcessorBuilder':
        self.logger = logger
        return self

    def set_config(self, config: Config) -> 'MatchupProcessorBuilder':
        self.config = config
        return self

    def build(self) -> MatchupProcessor:
        if not all([self.fetcher, self.parser, self.logger, self.config]):
            raise ValueError("All dependencies must be set before building")
        return MatchupProcessor(self.fetcher, self.parser, self.logger, self.config)

class DataParser:
    def __init__(self, categories: List[str]):
        self.categories = categories

    def parse_team_data(self, html: str) -> Dict[str, str]:
        soup = BeautifulSoup(html, 'html.parser')
        return self._parse_data(soup, self.categories)

    def _parse_data(self, soup: BeautifulSoup, categories: List[str]) -> Dict[str, str]:
        data = {}
        for category in categories:
            element = soup.find(text=category)
            if element:
                data[category] = element.text.strip()
            else:
                logging.warning(f"Failed to find element for category '{category}'")
        return data


@staticmethod
def parse_results(html: str, date: datetime) -> Dict:
    logging.info("Parsing results...")
    soup = BeautifulSoup(html, 'html.parser')

    results = {}

    for game in soup.find_all('div', class_='game_summary'):
        teams = game.find_all('tr')

        # Extract winner and loser
        winner = teams[1].find('a').text
        loser = teams[0].find('a').text

        # Extract scores
        winner_score = int(teams[1].find('td', class_='right').text)
        loser_score = int(teams[0].find('td', class_='right').text)

        # Check final status
        gamelink_element = teams[1].find('td', class_='right gamelink')
        if gamelink_element is not None:
            status = gamelink_element.span.text
        else:
            status = None  # or some default value

        if status != "Final":
            continue

        # Build matchup dict
        matchup = {
            'home': winner if winner_score > loser_score else loser, 
            'away': loser if winner_score > loser_score else winner,
            'date': date  # Add the date to the matchup dict
        }

        # Record winner and loser
        if matchup not in results:
            results[matchup] = {}
        results[matchup][winner] = 'winner'
        results[matchup][loser] = 'loser'

    logging.info("Parsing completed.")
    return results


def get_backtest_dates(
    days: int, 
    start: datetime = None, 
    end: datetime = None,
    date_format: str = "%Y%m%d"
) -> List[str]:

    # Validate inputs
    if not isinstance(days, int) or days < 1:
        raise ValueError("Days must be a positive integer")
  
    if start and not isinstance(start, datetime):
        raise ValueError("Start must be a datetime object")

    if end and not isinstance(end, datetime):
        raise ValueError("End must be a datetime object")

    # Calculate start and end dates
    end = end or datetime.now()
    start = start or end - timedelta(days=days-1)

    # Handle days with no matchups
    dates = []
    current = start
    while len(dates) < days:
        try:
            dates.append(current)
            current += timedelta(days=1)
        except NoMatchupsError:
            continue

    # Format dates
    return [date.strftime(date_format) for date in dates]


def validate_inputs(days: int, start: datetime, end: datetime) -> None:
    if not isinstance(days, int) or days < 1:
        raise ValueError("Days must be a positive integer")
  
    if start and not isinstance(start, datetime):
        raise ValueError("Start must be a datetime object")

    if end and not isinstance(end, datetime):
        raise ValueError("End must be a datetime object")
    
def validate_matchup_dict(matchup_dict):
    required_keys = ['date', 'home', 'away']
    for key in required_keys:
        if key not in matchup_dict:
            raise ValueError(f"Key '{key}' is missing from matchup_dict")
    

def calculate_dates(days: int, start: datetime, end: datetime = None) -> List[datetime]:
    end = end or datetime.now()  # Use provided end date or current date/time if end is None

    dates = []
    current = start
    while len(dates) < days:
        try:
            dates.append(current)
            current += timedelta(days=1)
        except NoMatchupsError:
            continue

    return dates


def handle_error(message: str):
    """Handle errors by logging and returning default values."""

    print(message)
    return {}


def calculate_score(home_team_stats: dict, away_team_stats: dict, config: Config) -> int:
    logging.info("Calculating score...")
    score = 0
    for stat_name, weight in config.rank_difference.items():
        rank_1 = home_team_stats.get(stat_name)
        rank_2 = away_team_stats.get(stat_name)
        if rank_1 is not None and rank_2 is not None:
            if rank_1 > 0 and rank_2 > 0:
                if rank_1 > rank_2:
                    score += weight
    logging.info("Score calculation completed.")
    return score

def calculate_win_rate(projections, actual_winners):
    """Calculates the win rate of the projections."""
    # Validate inputs
    if not projections or not actual_winners:
        return 0

    correct_predictions = 0
    for game, predicted_winner in projections.items():
        if predicted_winner == actual_winners[game]:
            correct_predictions += 1

    return correct_predictions / len(projections)

def valid_projection(projection, config):
  # Validate required keys
  required_keys = ['date', 'Home', 'Away', 'Projection']
  if not all(key in projection for key in required_keys):
    return False

  # Validate date format
  try:
    datetime.strptime(projection['date'], '%Y-%m-%d') 
  except ValueError:
    return False

  # Validate team names
  home_team = projection['Home']
  away_team = projection['Away']
  if home_team not in config.team_name_mapping or away_team not in config.team_name_mapping:
    return False

  # Validate projection
  if projection not in [home_team, away_team]:
    return False

  return True

def validate_projections(projections):
  """Validate list of projections"""
  
  # Check projections is a list
  if not isinstance(projections, list):
    return False

  # Check each projection is a dict with required keys
  for projection in projections:
    if not isinstance(projection, dict):
      return False
    required_keys = ['date', 'home', 'away', 'winner']
    if not all(key in projection for key in required_keys):
      return False

  return True

def projected_score(home_team_stats: dict, away_team_stats: dict, config: Config) -> int:
    """
    Calculate projected score for a matchup based on team stat ranks

    Args:
        home_team_stats (dict): Stats and ranks for home team
        away_team_stats (dict): Stats and ranks for away team
        config (Config): Config instance

    Returns:
        int: Projected score 
    """

    score = 0

    for stat_name, weight in config.rank_difference.items():
        
        home_rank = home_team_stats.get(stat_name)
        away_rank = away_team_stats.get(stat_name)
        
        if home_rank is not None and away_rank is not None:
            
            rank_diff = home_rank - away_rank
            
            if rank_diff > 0:
                score += rank_diff * weight

    return score



async def fetch_matchups(session, date, num_days):
    """Fetches matchups for a given date and number of days."""
    matchups = []
  
    for i in range(num_days):
        current_date = date - timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
      
        url = f"https://www.teamrankings.com/mlb/schedules/?date={date_str}"
        logging.info(f"Visiting URL: {url}")  # Log the URL

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    matchups.extend(await extract_matchups(html))
                else:
                    logging.warning("Error fetching matchups for %s, status %s", date_str, response.status)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logging.warning(f"Page not found: {url}")
            else:
                raise ClientError(f"Error fetching {url}: {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error fetching matchups for {date_str}: {e}") from e

    return matchups


async def fetch_results(date):
  """Fetches results for a given date."""
  
  date_str = date.strftime("%Y/%m/%d")
  
  url = f"https://www.baseball-reference.com/boxes/?month={date.month}&day={date.day}&year={date.year}"
  logging.info(f"Visiting URL: {url}")  # Log the URL

  async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
      html = await response.text()

  # Parse HTML and extract results  
  results = DataParser.parse_results(html)

  return results

async def get_results_for_date_range(start_date: str, days: int) -> Results:
    logging.info(f"Getting results for date range: {start_date} to {start_date + timedelta(days=days)}")
    results = {}

    # Create a list of tasks to fetch results for each date
    tasks = [fetch_results(start_date + timedelta(days=i)) for i in range(days)]

    # Use asyncio.gather to run the tasks concurrently
    results_list = await asyncio.gather(*tasks)

    # Combine the results into a single dictionary
    for i, result in enumerate(results_list):
        date = start_date + timedelta(days=i)
        results[date] = result

    return results

def get_single_game_boxscore(projection: Projection, actual_winner_dict: Results) -> str:
    assert "Home" in projection and "Away" in projection and "Projection" in projection, "Invalid projection data"
    assert projection["date"] in actual_winner_dict, "No actual results for the given date"

    home_team = projection["Home"]
    away_team = projection["Away"]
    projected_winner = projection["Projection"]  

    actual_game_results = actual_winner_dict[projection["date"]]
  
    return actual_game_results[home_team] if projected_winner == home_team else actual_game_results[away_team]

def compare_winners(matchups, projections):

  # Group matchups and projections by date
  matchup_by_date = defaultdict(list)
  projections_by_date = defaultdict(list)
  
  for matchup in matchups:
    matchup_by_date[matchup.date].append(matchup)
  
  for projection in projections:  
    projections_by_date[projection.date].append(projection)

  # Initialize accuracy tracking
  total = 0
  correct = 0 
  
  # Iterate through dates
  for date in matchup_by_date:

    matchups = matchup_by_date[date]
    projections = projections_by_date[date]
    
    # Get projected winners
    projected_winners = {p.team for p in projections}
    
    # Check each matchup
    for matchup in matchups:
    
      winner = matchup.winner
      
      if winner in projected_winners:
        correct += 1
        
      total += 1

  # Calculate accuracy  
  accuracy = correct / total
  
  return accuracy

async def extract_matchups(html: str) -> List[Matchup]:
    logging.info("Extracting matchups...")
    soup = BeautifulSoup(html, 'html.parser')
    cells = soup.find_all("td", class_="text-left nowrap")

    return [
        {
            "matchup": f"{home_name} at {away_name}",
            "home": home_name,
            "away": away_name,
            "date": datetime.now()  # Add the current date to the matchup dict
        }
        for cell in cells
        if cell.find("a") and cell.find("a")["href"].startswith("/mlb/matchup/")
        if (match := re.search(r'#\d+\s+(.*?)\s+at\s+#\d+\s+(.*?)$', cell.text.strip()))
        if (home_name := match.group(1).lower()) and (away_name := match.group(2).lower())
    ]


def write_results(projections: List[Projection], actual_winner_dict: dict, start_date: str, days: int, output_file: str) -> None:
    logging.info("Writing results...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w') as f:
        f.write("Date,Home Team,Away Team,Projection,Actual Winner\n")
        
        for projection in projections:
            date = projection["date"]
            if date in actual_winner_dict and len(actual_winner_dict[date]) >= 2:
                actual_winner = get_single_game_boxscore(projection, actual_winner_dict)
                f.write(f"{date},{projection['Home']},{projection['Away']},{projection['Projection']},{actual_winner}\n")

    logging.info("Writing completed.")


def write_projections(projections, filename):
  """Write projections to a CSV file"""

  # Validate projections
  if not validate_projections(projections):
    raise ValueError('Invalid projections list')

  # Convert projections to namedtuple
  projections = [Projection(**p) for p in projections]

  # Open file and create writer
  with open(filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=Projection._fields)

    # Write header
    writer.writeheader()

    # Write projections as rows
    writer.writerows(projections)


async def run_backtester(session, date, period, output_file):

  logging.info("Starting backtest for %s over %d days", date, period)

  try:
    # Get backtest dates
    backtest_dates = get_backtest_dates(period, start=date)

    # Get projected matchups
    matchups = []
    for date in backtest_dates:
        matchups.extend(await fetch_matchups(session, date, 1))
    
    # Generate projections
    projected = []
    for matchup in matchups:
      projected.append(projected_score(matchup))
    
    # Write projected matchups
    await write_projections(projected, 'projected.csv')

    # Get actual results
    actual = await get_results_for_date_range(date, period)

    # Combine projected and actual results
    combined = matchups + actual

    # Calculate win rate
    win_rate = calculate_win_rate(combined)
    logging.info("Win rate: %.2f%%", win_rate*100)

    # Write final results
    await write_results(combined, output_file)

  except Exception as e:
    logging.exception("Error in backtest: %s", e)

  finally:
    await session.close()

  return matchups



async def get_session():
    # Create an aiohttp ClientSession as an async context manager
    async with aiohttp.ClientSession() as session:
        yield session


async def main(days: int, config_path: str = 'config.ini'):
    """
    Main function to get and process matchups.

    Args:
        days (int): Number of days to backtest.
        config_path (str): Path to the configuration file. Defaults to 'config.ini'.
    """

    # Validate input
    if not days:
        raise ValueError("Days is required")

    # Log the received arguments
    logging.info(f"Main received: days={days}")

    # Create an instance of Config
    config = Config.from_file(config_path)

    # Get today's date
    today = datetime.today()

    # Create an instance of Fetcher, Parser, and MatchupProcessor
    fetcher = Fetcher(config)
    parser = Parser(config.stats_categories)
    processor = MatchupProcessor(fetcher, parser)

    # Get the dates to backtest
    dates = calculate_dates(days, start=today)

    # Create a list to store the results
    results = []

    # Create an aiohttp ClientSession
    async with aiohttp.ClientSession() as session:

        # Process each date
        for date in dates:
            # Get the matchups for the date
            matchup_dicts = await fetch_matchups(session, date, days)

            # Convert the dictionaries to Matchup objects
            matchups = [Matchup(home=Team(matchup_dict['home']), 
                                away=Team(matchup_dict['away']), 
                                matchup=matchup_dict['matchup'], 
                                date=matchup_dict['date']) for matchup_dict in matchup_dicts]

            # Process each matchup
            for matchup in matchups:
                result = await processor.process_matchup(session, matchup)
                results.append(result)

    # Write the results to a CSV file
    write_results(results, 'results.csv')

    # Print the results
    for result in results:
        print(result)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Run the backtest.")

    # Add the arguments
    parser.add_argument('days', type=int, nargs='?', default=7, help='The backtest period in days. Default is 7.')
    parser.add_argument('--config', type=str, default='config.ini', help='Path to the configuration file. Default is config.ini.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    asyncio.run(main(args.days, args.config))
