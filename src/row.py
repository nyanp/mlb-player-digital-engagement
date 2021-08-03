from typing import List, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Row:
    date: Any
    engagement: List
    games: List
    rosters: List
    player_box_scores: List
    team_box_scores: List
    transactions: List
    standings: List
    awards: List
    events: List
    player_twitter_followers: List
    team_twitter_followers: List
