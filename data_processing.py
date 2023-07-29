from typing import List, Dict, Any
from collections import defaultdict

def process_matchups(matchups: List[Dict[str, Any]], stats_categories: List[str]) -> List[Dict[str, Any]]:
    # Initialize a list to store the processed matchups
    processed_matchups = []

    # Iterate over each matchup
    for matchup in matchups:
        # Get the home and away teams and their ranks
        home_team = matchup['home']
        away_team = matchup['away']
        home_ranks = matchup['home_ranks']
        away_ranks = matchup['away_ranks']

        # Initialize a dictionary to store the processed matchup
        processed_matchup = {
            'home': home_team,
            'away': away_team,
            'home_ranks': {},
            'away_ranks': {}
        }

        # Iterate over each stat category
        for stat_category in stats_categories:
            # Get the home and away ranks for the stat category
            home_rank = home_ranks.get(stat_category, None)
            away_rank = away_ranks.get(stat_category, None)

            # Check if the home and away ranks were found
            if home_rank is not None and away_rank is not None:
                # Calculate the rank difference
                rank_difference = home_rank - away_rank

                # Add the rank difference to the processed matchup
                processed_matchup['home_ranks'][stat_category] = rank_difference
                processed_matchup['away_ranks'][stat_category] = -rank_difference

        # Add the processed matchup to the processed matchups list
        processed_matchups.append(processed_matchup)

    return processed_matchups

def run_backtester(matchups: List[Dict[str, Any]], stats_categories: List[str]) -> Dict[str, Any]:
    # Initialize a dictionary to store the backtester results
    backtester_results = {}

    # Iterate over each matchup
    for matchup in matchups:
        # Get the home and away teams and their ranks
        home_team = matchup['home']
        away_team = matchup['away']
        home_ranks = matchup['home_ranks']
        away_ranks = matchup['away_ranks']

        # Initialize a dictionary to store the backtester result for the matchup
        backtester_result = {
            'home': home_team,
            'away': away_team,
            'home_ranks': {},
            'away_ranks': {}
        }

        # Iterate over each stat category
        for stat_category in stats_categories:
            # Get the home and away ranks for the stat category
            home_rank = home_ranks.get(stat_category, None)
            away_rank = away_ranks.get(stat_category, None)

            # Check if the home and away ranks were found
            if home_rank is not None and away_rank is not None:
                # Calculate the rank difference
                rank_difference = home_rank - away_rank

                # Add the rank difference to the backtester result
                backtester_result['home_ranks'][stat_category] = rank_difference
                backtester_result['away_ranks'][stat_category] = -rank_difference

        # Add the backtester result to the backtester results dictionary
        backtester_results[(home_team, away_team)] = backtester_result

    return backtester_results

def calculate_score(matchup: Dict[str, Any], rank_difference: Dict[str, int]) -> int:
    # Get the home and away ranks
    home_ranks = matchup['home_ranks']
    away_ranks = matchup['away_ranks']

    # Initialize the score
    score = 0

    # Iterate over each stat category
    for stat_category, rank_diff in rank_difference.items():
        # Get the home and away ranks for the stat category
        home_rank = home_ranks.get(stat_category, None)
        away_rank = away_ranks.get(stat_category, None)

        # Check if the home and away ranks were found
        if home_rank is not None and away_rank is not None:
            # Calculate the score
            score += (home_rank - away_rank) * rank_diff

    return score

def calculate_win_rate(backtester_results: Dict[str, Any], actual_winners: Dict[str, str]) -> float:
    # Initialize the number of correct predictions and the total number of predictions
    num_correct_predictions = 0
    num_total_predictions = 0

    # Iterate over each backtester result
    for matchup, backtester_result in backtester_results.items():
        # Get the home and away teams
        home_team = backtester_result['home']
        away_team = backtester_result['away']

        # Get the actual winner
        actual_winner = actual_winners.get((home_team, away_team), None)

        # Check if the actual winner was found
        if actual_winner is not None:
            # Get the predicted winner
            predicted_winner = home_team if backtester_result['score'] > 0 else away_team

            # Check if the prediction was correct
            if predicted_winner == actual_winner:
                num_correct_predictions += 1

            num_total_predictions += 1

    # Calculate the win rate
    win_rate = num_correct_predictions / num_total_predictions if num_total_predictions > 0 else 0

    return win_rate

def group_matchups_by_date(matchups: List[Dict[str, Any]]) -> Dict[datetime, List[Dict[str, Any]]]:
    # Initialize a dictionary to store the matchups grouped by date
    matchups_by_date = defaultdict(list)

    # Iterate over each matchup
    for matchup in matchups:
        # Get the date
        date = matchup['date']

        # Add the matchup to the list of matchups for the date
        matchups_by_date[date].append(matchup)

    return matchups_by_date
