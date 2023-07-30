import sys
import argparse
import asyncio
import aiohttp
from aiohttp import ClientSession
from asyncio import Semaphore
from typing import List, Dict, Any
import configparser
from configparser import ConfigParser
import logging
from bs4 import BeautifulSoup
import csv
from collections import defaultdict
from datetime import datetime, timedelta
import re
from dateutil.relativedelta import relativedelta
from collections import namedtuple
import requests
import traceback
import time
from urllib.parse import quote_plus
from aiohttp import ClientTimeout
from cachetools import cached, LRUCache, TTLCache
from html.parser import HTMLParseError

html_cache = TTLCache(maxsize=100, ttl=300)

# Create a logger
logger = logging.getLogger(__name__)

# Set up basic configuration for the logger
logging.basicConfig(filename='logfile.log', level=logging.INFO)

# Set the log level for the part of the code that prints HTML data to DEBUG
logging.getLogger("asyncio").setLevel(logging.ERROR)  # Ignore asyncio logging (optional)
logging.getLogger("aiohttp.client").setLevel(logging.ERROR)  # Ignore aiohttp client logging (optional)
logging.getLogger("aiohttp.internal").setLevel(logging.ERROR)  # Ignore aiohttp internal logging (optional)
logging.getLogger("chardet.charsetprober").setLevel(logging.ERROR)  # Ignore chardet logging (optional)

# Set the log level for the HTML data printing to DEBUG
logger.setLevel(logging.ERROR)  # Use logger.setLevel(logging.WARNING) if you want to see warning messages

# Load the config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

categories = config['Stats']['categories'].split(', ')
weights = [int(x) for x in config['Stats']['weights'].split(', ')]

default_weights = dict(zip(categories, weights)) 

# Get the valid team names from the config file
valid_teams = list(config['team_name_mapping'].keys())

# Define constants
RANKING_MAX = 31
DIV = 'div'
DATA_CONTAINER = 'data-container'
TEAM_BASE_URL = "https://www.teamrankings.com/mlb/team/"
MAX_RETRIES = 5
DEFAULT_TIMEOUT = 10

HashableMatchup = namedtuple('HashableMatchup', ['home', 'away'])

# Define a separate cache dictionary
data_cache = {}
result_cache = {}
matchups_cache = {}

# Define a semaphore to limit concurrent requests
semaphore = Semaphore(10)

# Define data containers
team_data = {}
projections = []
all_team_data = []

# Initialize the actual_winner_dict dictionary
actual_winner_dict = {}

# Dictionary to map team names to URL-friendly names
team_name_mapping = {}
for team_name in config.options('team_name_mapping'):
    url_friendly_name = config.get('team_name_mapping', team_name)
    team_name_mapping[team_name] = url_friendly_name.lower()  # Convert to lowercase here

# Dictionary to map stats categories to their corresponding rank
stats_categories = {}
for idx, category in config.items('stats_categories'):
    stats_categories[idx] = category

# Dictionary to map stats categories to their rank difference
rank_difference = {}
for stat_name, rank_diff in config.items('rank_difference'):
    rank_difference[stat_name] = int(rank_diff)

# Define data keys
TEAM_DATA_KEY = "team_data"
MATCHUP_KEY = "matchup"

# Define a namedtuple for a matchup
Matchup = namedtuple('Matchup', ['home', 'away'])

class Config:

  def __init__(self, filepath):
    self.parser = configparser.ConfigParser()
    self.parser.read(filepath)

    required_sections = ["team_name_mapping", "stats_categories"]
    for section in required_sections:
      if not self.parser.has_section(section):
        raise Exception(f"Section '{section}' not found in config file")

    self._validate_config()
    
  def _validate_config(self):
    # Input validation
    pass

  @property
  def teams(self):
    return self.parser["team_name_mapping"]

  @property 
  def team_map(self):
    return {team: self.teams[team] for team in self.teams}

  @property
  def categories(self):
    return dict(self.parser["stats_categories"])

  @property
  def log_level(self):
    return self.parser.get("logging", "level")

class CachingClientSession(ClientSession):

  def __init__(self, *args, cache_size=100, **kwargs):
    super().__init__(*args, **kwargs)
    self.cache = LRUCache(cache_size)
    self.logger = logging.getLogger(__name__)

  async def fetch(self, url, **kwargs):
    if url in self.cache:
      self.logger.info(f"Fetching {url} from cache")
      return self.cache[url]
    
    try:
      resp = await super().get(url, **kwargs)
      if resp.status == 200:
        html = await resp.text()
        self.cache[url] = html
        return html
      else:
        self.logger.warning(f"Failed to fetch {url}, status {resp.status}")
    except ClientError as e:
      self.logger.error(f"Error fetching {url}: {e}")

  async def close(self):
    await super().close()
    self.cache.clear()


def get_team_url(team_name):
  normalized = team_name.lower().replace(" ", "-")
  slug = team_name_mapping.get(normalized, normalized)
  encoded_slug = quote_plus(slug)
  
  return f"{TEAM_BASE_URL}/{encoded_slug}/stats"


async def fetch_team_page(session, url, timeout=DEFAULT_TIMEOUT):

  try:
    timeout = ClientTimeout(total=timeout)
    async with session.get(url, timeout=timeout) as resp:
      resp.raise_for_status()
      return await resp.text()

  except ClientError as e:
    logger.error(f"Error fetching {url}: {e}")
    return ""

async def collect_team_html(cache, session, team_url, retries=3):
  
  url = team_url
  for retry in range(retries):
    html = cache.get(url)
    if html is None: 
      logger.info(f"Fetching team page: {url}")
      html = await fetch_team_page(session, url)
      if html:
        cache[url] = html

    if html:
      logger.info(f"Received {len(html)} chars for {url}")
      return html

  logger.warning(f"No HTML received for {url} after {retries} retries")
  return ""  


class TeamStatsParser:

  CATEGORIES = ["batting", "pitching", "fielding"]
  
  def __init__(self, html):
    self.soup = BeautifulSoup(html, 'html.parser')
    self.cache = LRUCache(maxsize=3)

  @cached(cache=cache)
  def parse_ranks(self):
    ranks = {}

    for category in self.CATEGORIES:
      rank = self._extract_rank(category)
      if rank:
        ranks[category] = rank

    return ranks

  def _extract_rank(self, category):
    if category not in self.CATEGORIES:
      return

    category_td = self._find_category_td(category)
    if not category_td:
      return

    rank_td = self._find_rank_td(category_td)
    if not rank_td:
      return

    rank = self._parse_rank(rank_td)
    return rank

  def _find_category_td(self, category):
    return self.soup.find('td', string=category)

  def _find_rank_td(self, category_td):
    return category_td.find_next_sibling('td', class_='nowrap', style='text-align:right;')
    
  def _parse_rank(self, rank_td):
    return rank_td.small.text.strip()


def fetch_and_parse(url, categories, max_retries=3):

  for retry in range(max_retries):
    try:
      resp = requests.get(url)
      resp.raise_for_status()

      soup = BeautifulSoup(resp.text, 'html.parser')
      data = parse_data(soup, categories)
      return data

    except RequestException as e:
      logger.error(f"Error fetching {url}: {e}")

  return {}

def parse_data(soup, categories):
  
  data = {}

  for cat in categories:
    cat_td = soup.find('td', string=cat.lower())
    if cat_td:
      rank_td = cat_td.find_next_sibling('td', class_='nowrap', style='text-align:right;')
      if rank_td:
        rank = rank_td.small.text.strip()
        data[cat] = rank

  return data


async def fetch_team_data(session, team_url):
  async with session.get(team_url) as resp:
    return await resp.text()

async def extract_team_ranks(html, categories):
  # parsing logic...
  return ranks

async def get_team_data(session, team, categories):
  url = get_team_url(team)
  html = await fetch_team_data(session, url)
  return extract_team_ranks(html, categories)

async def process_matchup(session, matchup, categories):

  home_team = matchup['home']
  away_team = matchup['away']
  
  home_data = await get_team_data(session, home_team, categories)
  away_data = await get_team_data(session, away_team, categories)

  return {
    home_team: home_data,
    away_team: away_data
  }

async def fetch_and_extract(session, matchup, categories, queue):

  try:
    data = await process_matchup(session, matchup, categories)
    await queue.put((data, 'matchup'))
    return data
  
  except ClientError as e:
    logger.error(f"Error processing {matchup}: {e}")
    raise

  except Exception as e:
    logger.exception(f"Error processing {matchup}: {e}")
    raise


@lru_cache(maxsize=32)
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
  start = start or end - relativedelta(days=days-1)

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

async def fetch_matchups(
    session: ClientSession,
    date: datetime
) -> List[Dict]:

  date_str = date.strftime("%Y-%m-%d")

  if date_str in cache:
    return cache[date_str]

  try:
    matchups = await get_matchups(session, date_str)

    # Validate each matchup
    valid_matchups = []
    for matchup in matchups:
      if validate_matchup(matchup):
        valid_matchups.append(matchup)
      else:
        log_invalid(matchup)
    
    if not valid_matchups:
      raise NoDataError(f"No valid matchups for {date_str}")

    cache[date_str] = valid_matchups
    return cache[date_str]

  except NoDataError as e:
    log_error(e)
    return [] # Return empty list

  except ClientError as e:
    log_error(f"Error fetching {date_str}: {e}")
    raise

  except Exception as e:
    log_exception(f"Error processing {date_str}: {e}")
    raise

def validate_html(soup):
  # Check for required elements
  return True

@cached(html_cache)  
def parse_html(html):
  try:
    soup = BeautifulSoup(html, 'html.parser')
    if not validate_html(soup):
      return None
    return soup
  except HTMLParseError as e:
    logger.error("Error parsing HTML: %s", e)
    return None

def parse_matchup(cell):
  # logic to parse matchup 
  return parsed_matchup

def validate_matchup(matchup):
  # Validate required fields
  return is_valid

def extract_matchups(html):

  soup = parse_html(html)
  if not soup:
    return []

  matchups = []

  for cell in soup.find_all('td'):
    parsed = parse_matchup(cell)
    if parsed:
      if validate_matchup(parsed):
        matchups.append(parsed)
        logger.info("Valid matchup: %s", parsed)
      else:
        logger.warning("Invalid matchup: %s", parsed)

  return matchups

async def run_backtester(session, date, period, output_file):

  logger.info("Starting backtest for %s over %d days", date, period)

  try:
    matchups = await get_matchups(session, date, period)
    
    # Generate projections
    projected = []
    for matchup in matchups:
      projected.append(generate_projection(matchup))
    
    # Write projected matchups
    write_csv(projected, 'projected.csv')

    # Get actual results
    actual = get_actual_results(date, period)

    # Combine
    combined = matchups + actual

    # Calculate win rate
    win_rate = calculate_win_rate(combined)
    logger.info("Win rate: %.2f%%", win_rate*100)

    # Write final results
    write_csv(combined, output_file)

  except Exception as e:
    logger.exception("Error in backtest: %s", e)

  finally:
    await session.close()

  return matchups

async def get_matchups(session, start_date, days):
  
  matchups = []

  for i in range(days):
    date = start_date + timedelta(days=i)  
    matchups.extend(await fetch_matchups(session, date))

  return matchups

def generate_projection(matchup):

  # Get home and away team stats
  home_stats = get_team_stats(matchup['home']) 
  away_stats = get_team_stats(matchup['away'])

  # Calculate score
  score = calculate_score(home_stats, away_stats, default_weights)

  # Determine projected winner based on score
  if score > 0:
    projected_winner = matchup['home']
  else:
    projected_winner = matchup['away']

  # Return projected matchup with winner
  return {
    'home': matchup['home'], 
    'away': matchup['away'],
    'projected_winner': projected_winner
  }


def get_actual_results(start_date, days):
  
  results = []

  for i in range(days):
    date = start_date + timedelta(days=i)
    results.extend(lookup_actual_results(date))
  
  return results

def calculate_win_rate(matchups):
  
  wins = 0
  total = len(matchups)

  for m in matchups:
    if m['projected_winner'] == m['actual_winner']:
      wins += 1

  return wins / total

def write_csv(rows, file):

  with open(file, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys()) 
    writer.writeheader()
    writer.writerows(rows)


async def process_matchups(matchups, stats, teams):

  logger.info(f"Processing {len(matchups)} matchups")

  # Fetch ranks in parallel
  queue = asyncio.Queue()
  tasks = [
    fetch_ranks(m, stats, queue, teams)
    for m in matchups
  ]

  try:
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for task in pending:
      task.cancel()
    if pending:
      raise Exception("Error fetching ranks")

  except Exception as e:
    logger.exception("Error processing matchups: %s", e)
    return []

  # Build rank dicts
  home_ranks = {}
  away_ranks = {}

  while not queue.empty():
    try:
      ranks, key = queue.get_nowait()  
    except Exception as e:
      logger.error("Error getting result from queue: %s", e)
      continue

    if key == 'home':
      home_ranks.update(ranks)
    elif key == 'away':
      away_ranks.update(ranks)

  # Enrich matchups
  enriched = []
  for m in matchups:
    if m['home'] in home_ranks and m['away'] in away_ranks:
      enriched.append({
        'matchup': m,
        'home_ranks': home_ranks[m['home']],
        'away_ranks': away_ranks[m['away']],
      })


async def get_all_matchups(session, current_date, days):

  logger.info("Fetching matchups for %s over %d days", current_date, days)

  # Limit concurrency
  sem = asyncio.Semaphore(10)

  # Cache matchups
  cache = {}

  tasks = []
  for i in range(days):
    date = current_date - timedelta(days=i)
    tasks.append(
      fetch_matchups(session, date, sem, cache)  
    )

  matchups = []
  for result in asyncio.as_completed(tasks):
    try:
      for matchup in await result:
        if validate_matchup(matchup):
          matchups.append(matchup)
    except Exception as e:
      logger.error("Error fetching matchups: %s", e)

  return matchups  

async def fetch_matchups(session, date, sem, cache):

  async with sem:
    if date in cache:
      return cache[date]

    matchups = await session.fetch(date)  
    cache[date] = matchups
    return matchups

def validate_matchup(matchup):
  # Validate matchup dict
  return True


@asynccontextmanager
async def timeit(func):
  start = time.time()
  try:
    yield
  finally:
    end = time.time()
    logger.info(f"{func.__name__} took {end-start:.2f} sec")


def get_winners(html):

  soup = BeautifulSoup(html, 'html.parser')

  winners = {}

  for game in soup.find_all('div', class_='game_summary'):

    teams = game.find_all('tr')

    # Extract winner and loser
    winner = teams[1].find('a').text
    loser = teams[0].find('a').text

    # Extract scores 
    winner_score = int(teams[1].find('td', class_='right').text)
    loser_score = int(teams[0].find('td', class_='right').text)

    # Check final status
    status = teams[1].find('td', class_='right gamelink').span.text
    if status != "Final":
      continue

    # Build matchup dict
    matchup = {
      'home': winner if winner_score > loser_score else loser,
      'away': loser if winner_score > loser_score else winner
    }

    # Record winner and loser
    if matchup not in winners:
      winners[matchup] = {}
    winners[matchup][winner] = 'winner'  
    winners[matchup][loser] = 'loser'

  return winners


def calculate_score(home_team_stats: dict, away_team_stats: dict) -> int:
    logger.info("Calculating score...")
    score = 0
    for stat_name, weight in rank_difference.items():
        rank_1 = home_team_stats.get(stat_name)
        rank_2 = away_team_stats.get(stat_name)
        if rank_1 is not None and rank_2 is not None:
            if rank_1 > 0 and rank_2 > 0:
                if rank_1 > rank_2:
                    score += weight
    logger.info("Score calculation completed.")
    return score


import logging

logger = logging.getLogger('win_rate')

def calculate_win_rate(projections, actual_winners):

  if not projections:
    logger.error('No projections provided')
    return 0

  if not actual_winners:
    logger.error('No actual winners provided') 
    return 0

  total = len(projections)
  correct = 0

  for projection in projections:

    if not valid_projection(projection):
      logger.error(f'Invalid projection: {projection}')
      continue

    date = projection['date']
    home = projection['Home']
    away = projection['Away']
    projected = projection['Projection']

    if date not in actual_winners:
      logger.error(f'No actual winner for {date}')
      continue

    if home not in actual_winners[date] or away not in actual_winners[date]:
      logger.error(f'Missing team for {date}')
      continue

    actual = actual_winners[date][home] if projected == home else actual_winners[date][away]
      
    if actual == projected:
      correct += 1

  if total == 0:
    return 0
  
  return correct / total

def valid_projection(projection):
  # Validate projection 
  return True

def validate_projections(projections):
  # validate projections
  return True
\
def write_csv(projections, file):

  if not validate_projections(projections):
    raise ValueError('Invalid projections')

  projections = [Projection(**p) for p in projections]

  try:
    with open(file, 'w') as f:
      writer = csv.DictWriter(f, fieldnames=Projection._fields)  
      writer.writeheader()
      writer.writerows(projections)

  except Exception as e:
    logger.exception(f"Error writing CSV: {e}")
    raise


async def run_script():

  parser = argparse.ArgumentParser()
  parser.add_argument('--days', type=int, required=True)
  parser.add_argument('--date', type=str)  
  parser.add_argument('--output', type=str, required=True)

  args = parser.parse_args()

  if args.days < 1:
    raise ValueError('Days must be greater than 0')

  date = args.date or datetime.now().strftime('%Y-%m-%d')

  try:
    winners = {}

    async with ClientSession() as session:
      await backtest(session, date, args.days, args.output, winners)

  except Exception as e:
    logger.exception(f'Error: {e}')
    print(traceback.format_exc())
    raise
  

async def backtest(session, date, days, output, winners):
    try:
        # Backtest logic
        pass  # Replace this with your actual code
    except Exception as e:
        # Log error details
        logger.error(f'Backtest failed: {e}')
        
        # Log traceback for debugging
        logger.debug(traceback.format_exc())

        # Re-raise exception
        raise


def validate_matchups(matchups):
  # Input validation
  pass

def group_by_date(matchups):

  if not validate_matchups(matchups):
    raise ValueError('Invalid matchups')

  groups = defaultdict(list)

  for matchup in matchups:

    if 'date' not in matchup:
      raise ValueError('Matchup missing date')

    groups[matchup['date']].append(matchup)

  return OrderedDict(sorted(groups.items()))

if __name__ == "__main__":
    asyncio.run(run_script())
