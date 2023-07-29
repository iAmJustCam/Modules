from typing import Dict, Any
from bs4 import BeautifulSoup
import re

def extract_ranks(html_content: str, stats_categories: List[str]) -> Dict[str, Any]:
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table with the stats data
    stats_table = soup.find('table', {'class': 'tr-table datatable scrollable'})

    # Initialize a dictionary to store the ranks
    ranks = {}

    # Check if the stats table was found
    if stats_table is not None:
        # Get all the rows in the table
        rows = stats_table.find_all('tr')

        # Iterate over each row
        for row in rows:
            # Get all the cells in the row
            cells = row.find_all('td')

            # Check if the row has the correct number of cells
            if len(cells) == 5:
                # Get the stat category and rank
                stat_category = cells[0].text.strip()
                rank = cells[1].text.strip()

                # Check if the stat category is in the list of stats categories
                if stat_category in stats_categories:
                    # Add the rank to the ranks dictionary
                    ranks[stat_category] = rank

    return ranks

def extract_matchups(html_content: str) -> List[Dict[str, str]]:
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table with the matchups data
    matchups_table = soup.find('table', {'class': 'tr-table datatable scrollable'})

    # Initialize a list to store the matchups
    matchups = []

    # Check if the matchups table was found
    if matchups_table is not None:
        # Get all the rows in the table
        rows = matchups_table.find_all('tr')

        # Iterate over each row
        for row in rows:
            # Get all the cells in the row
            cells = row.find_all('td')

            # Check if the row has the correct number of cells
            if len(cells) == 3:
                # Get the home and away teams
                home_team = cells[0].text.strip()
                away_team = cells[2].text.strip()

                # Add the matchup to the matchups list
                matchups.append({
                    'home': home_team,
                    'away': away_team
                })

    return matchups

def actual_winner(matchup: Dict[str, str], date: datetime) -> str:
    # Get the home and away teams
    home_team = matchup['home']
    away_team = matchup['away']

    # Get the actual winner from the actual_winner_dict dictionary
    actual_winner = actual_winner_dict.get((home_team, away_team, date), None)

    return actual_winner
