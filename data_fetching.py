import aiohttp
from aiohttp import ClientSession
from asyncio import Semaphore
from typing import List, Dict, Any
import logging
from bs4 import BeautifulSoup
import re
import configparser
from configparser import ConfigParser
import hashlib
import json
from collections import namedtuple, defaultdict

# Assuming you have already defined the logger
logger = logging.getLogger(__name__)

# Define constants
RANKING_MAX = 31
DIV = 'div'
DATA_CONTAINER = 'data-container'
TEAM_BASE_URL = "https://www.teamrankings.com/mlb/team/"

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

# Load the config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# Get the valid team names from the config file
valid_teams = list(config['team_name_mapping'].keys())

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

# Create a reverse mapping from "short" team names to "long" ones
reverse_team_name_mapping = {v: k for k, v in team_name_mapping.items()}

# Define a namedtuple for a matchup
Matchup = namedtuple('Matchup', ['home', 'away'])

class CachingClientSession(aiohttp.ClientSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self.logger = logging.getLogger(__name__)

    async def fetch_page(self, url, cookies={}):
        # Check the cache before making a request
        if url in self._cache:
            self.logger.info(f"Fetching page from cache: {url}")
            return self._cache[url]

        # If the page is not in the cache, make a request to fetch the page
        try:
            self.logger.info(f"Fetching page from URL: {url}")
            async with self.get(url, cookies=cookies) as response:
                if response.status == 200:
                    html_content = await response.text()
                    self._cache[url] = html_content  # Update the cache
                    return html_content
                else:
                    self.logger.warning(f"Failed to fetch page from URL: {url}. Status code: {response.status}")
                    return None
        except aiohttp.ClientError as e:
            self.logger.error(f"AIOHTTP ClientError occurred while fetching page from URL: {url}: {e}")
            return None

def get_team_url(team_name: str) -> str:
    team_slug = team_name_mapping.get(team_name.lower(), team_name.lower().replace(' ', '-'))
    team_url = f"https://www.teamrankings.com/mlb/team/{team_slug}/stats"
    return team_url

async def collect_html_from_team_page(session: aiohttp.ClientSession, team_url: str) -> str:
    print(f"Starting to fetch HTML content for team URL '{team_url}'")  # Debugging print statement
    try:
        async with session.get(team_url) as response:
            print(f"Received response for team URL '{team_url}'")  # Debugging print statement
            response.raise_for_status()  # Check if the request was successful
            print(f"Response for team URL '{team_url}' was successful")  # Debugging print statement
            html_content = await response.text()
            print(f"HTML content for team URL '{team_url}' has been fetched")  # Debugging print statement
            return html_content
    except Exception as e:
        print(f"Error fetching HTML content for team URL '{team_url}': {e}")
        return ""

async def fetch_matchups(session: CachingClientSession, date: datetime) -> List[Dict[str, str]]:
    date_str = date.strftime('%Y-%m-%d')
    if date_str in matchups_cache:
        logger.info(f"Fetching matchups from cache for date: {date_str}")
        return matchups_cache[date_str]

    matchups_list = []
    try:
        url = f"https://www.teamrankings.com/mlb/schedules/?date={date_str}"
        logger.info(f'Visiting URL: {url}')
        
        # Use cached HTML content if available
        async with session.get(url) as response:
            html = await response.text()

        matchups = await extract_matchups(html)
        logger.info(f"Extracted matchups: {matchups}")
        for matchup in matchups:
            if isinstance(matchup, dict) and 'matchup' in matchup:
                logger.debug(f"Got matchup home: {matchup['home']}, away: {matchup['away']}")
            else:
                logger.error(f"Unexpected matchup format: {matchup}")
        matchups_list.extend(matchups)

        # Cache the result
        matchups_cache[date_str] = matchups_list

        return matchups_list
    except aiohttp.ClientError as e:
        logger.error(f"AIOHTTP ClientError occurred while fetching matchups: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred while fetching and extracting matchups: {e}")
        raise

async def fetch_and_extract_team_ranks(
    session: aiohttp.ClientSession,
    matchup: Dict[str, str],
    stats_categories: List[str],
    queue: asyncio.Queue,
    valid_teams: List[str]
) -> Dict[str, Dict[str, Any]]:
    MATCHUP_KEY = "matchup"

    home_team = None  # Initialize with default value
    away_team = None  # Initialize with default value

    if 'home' in matchup:
        home_team = matchup['home'].lower()  # Convert to lowercase
    else:
        print("The 'home' key is missing from the matchup dictionary")  # Debugging print statement

    if 'away' in matchup:
        away_team = matchup['away'].lower()  # Convert to lowercase
    else:
        print("The 'away' key is missing from the matchup dictionary")  # Debugging print statement

    # Check if the home and away teams are valid
    if home_team not in valid_teams or away_team not in valid_teams:
        print(f"Invalid team(s) in matchup: home - {home_team}, away - {away_team}")  # Debugging print statement
        return {}

    # Get the team URLs
    home_team_url = get_team_url(home_team)
    away_team_url = get_team_url(away_team)

    # Fetch the HTML content for the home and away teams
    home_team_html = await collect_html_from_team_page(session, home_team_url)
    away_team_html = await collect_html_from_team_page(session, away_team_url)

    # Extract the ranks for the home and away teams
    home_team_ranks = extract_ranks(home_team_html, stats_categories)
    away_team_ranks = extract_ranks(away_team_html, stats_categories)

    # Create a dictionary to store the matchup data
    matchup_data = {
        MATCHUP_KEY: {
            'home': home_team,
            'away': away_team,
            'home_ranks': home_team_ranks,
            'away_ranks': away_team_ranks
        }
    }

    # Put the matchup data into the queue
    await queue.put(matchup_data)

    return matchup_data

async def get_all_matchups(session: aiohttp.ClientSession, date: datetime) -> List[Dict[str, Any]]:
    # Fetch the matchups for the given date
    matchups = await fetch_matchups(session, date)

    # Create a queue to store the matchup data
    queue = asyncio.Queue()

    # Fetch and extract the team ranks for each matchup
    tasks = [fetch_and_extract_team_ranks(session, matchup, stats_categories, queue, valid_teams) for matchup in matchups]

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Get all the matchup data from the queue
    all_matchup_data = []
    while not queue.empty():
        matchup_data = await queue.get()
        all_matchup_data.append(matchup_data)

    return all_matchup_data
