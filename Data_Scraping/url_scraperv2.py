import requests
from bs4 import BeautifulSoup

# For the Milwuakee Bucks 2021-22 season
BASE_URL = r"https://www.basketball-reference.com/teams/MIL/2022_games.html"


class URL_Scrape:

    Dates = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12
    }

    def __init__(self, base_url):
        self.r = requests.get(base_url)
        self.soup = BeautifulSoup(self.r.text, "lxml")
        self.rows = self.soup.find("table", id="games").tbody.findAll("tr")
        self.TABLE_HEAD_INDEXES = self._set_thead_indexes()

    def _set_thead_indexes(self):

        """returns a list of the indexes in the table of thead columns"""

        indexes = []
        for i, n in enumerate(self.rows): # replacing non data colums with None
            if len(n) != 15:
                indexes.append(i)

        return indexes

    def _convert_game_dates(self, game_dates):

        """converts plain text dates to a list of tuple in url param format"""

        formatted_dates = []
        for game in game_dates:
            date, year = [i.strip() for i in game.split(",")[1:]]
            month, day = date.split()
            formatted_dates.append((day, URL_Scrape.Dates[month], year))

        return formatted_dates

    def _get_game_dates(self):

        """scrapes the game dates for the whole teamn's season and returns a list of plain text dates"""

        game_dates = []

        for game in self.rows:
            link_tag = game.find("a")
            if link_tag: # the table headers have no links so are None types
                game_dates.append(link_tag.text)

        return game_dates

    def get_game_urls(self):

        """returns all the URLs for all the games in the regular season"""

        urls = []

        for game in self.rows:
            link_tag = game.find("td", attrs={"data-stat" : "box_score_text"})
            if link_tag:
                game_code = link_tag.a["href"].split("/")[2]
                urls.append(f"https://www.basketball-reference.com/boxscores/pbp/{game_code}")
        
        return urls

    def get_game_metadata(self):

        """returns a list of dictionaries containing metadata about every game in the regular season"""

        metadata = []

        dates = self._get_game_dates()
        count = 0 # to index the non-table rows

        for ind, game in enumerate(self.rows):

            if ind in self.TABLE_HEAD_INDEXES: # skip over table headers
                continue

            team_pts = game.find("td", attrs={"data-stat" : "pts"})
            opp_pts = game.find("td", attrs={"data-stat" : "opp_pts"})

            if team_pts and opp_pts: # discounting None values
                
                metadata_dict = {
                    "team points": team_pts.text,
                    "opposition points": opp_pts.text,
                    "date": dates[count]
                }


                metadata.append(metadata_dict)

            count += 1
            
        return metadata
