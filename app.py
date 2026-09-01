from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
from dotenv import load_dotenv

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

print("TMDB API KEY LOADED:",bool (TMDB_API_KEY)
)

# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "movies.csv"
)


# ==========================================
# LOAD DATASET
# ==========================================

try:

    movies = pd.read_csv(DATA_PATH)

except FileNotFoundError:

    raise FileNotFoundError(
        "movies.csv not found. "
        "Make sure it is inside the data folder."
    )


# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "title",
    "genres"
]

for column in required_columns:

    if column not in movies.columns:

        raise ValueError(
            f"Required column '{column}' "
            "is missing from movies.csv"
        )


# ==========================================
# DATA CLEANING
# ==========================================

movies = movies[
    ["title", "genres"]
].copy()

movies.dropna(
    subset=["title", "genres"],
    inplace=True
)

movies.drop_duplicates(
    subset="title",
    inplace=True
)

movies.reset_index(
    drop=True,
    inplace=True
)

movies["genres"] = movies["genres"].astype(str)

movies["genres"] = (
    movies["genres"]
    .str.replace("|", " ", regex=False)
    .str.lower()
)


# ==========================================
# COUNT VECTORIZER
# ==========================================

count = CountVectorizer(
    stop_words="english"
)

count_matrix = count.fit_transform(
    movies["genres"]
)


# ==========================================
# COSINE SIMILARITY
# ==========================================

cosine_sim = cosine_similarity(
    count_matrix,
    count_matrix,
    dense_output=False
)


# ==========================================
# TMDB POSTER FUNCTION
# ==========================================
def get_movie_poster(title):

    if not TMDB_API_KEY:
        print("ERROR: TMDB_API_KEY is missing.")
        return None

    try:

        import re

        # Clean movie title
        clean_title = title.strip()

        # Remove year from title
        title_without_year = re.sub(
            r"\s*\(\d{4}\)\s*$",
            "",
            clean_title
        ).strip()

        # Extract year
        year_match = re.search(
            r"\((\d{4})\)\s*$",
            clean_title
        )

        year = None

        if year_match:
            year = year_match.group(1)

        # TMDB URL
        url = "https://api.themoviedb.org/3/search/movie"

        params = {
            "api_key": TMDB_API_KEY,
            "query": title_without_year,
            "language": "en-US",
            "include_adult": "false"
        }

        if year:
            params["year"] = year

        print("Searching TMDB for:", title_without_year)

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        print("TMDB status:", response.status_code)

        if response.status_code != 200:
            print("TMDB response:", response.text)
            return None

        data = response.json()

        results = data.get("results", [])

        if not results:

            print(
                "No TMDB movie found:",
                title
            )

            return None

        # Find movie with poster
        for movie in results:

            poster_path = movie.get("poster_path")

            if poster_path:

                poster_url = (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

                print(
                    "Poster found:",
                    poster_url
                )

                return poster_url

        print(
            "Movie found but no poster:",
            title
        )

        return None

    except requests.exceptions.ConnectionError as e:

        print(
            "TMDB CONNECTION ERROR:",
            e
        )

        return None

    except requests.exceptions.Timeout:

        print(
            "TMDB REQUEST TIMED OUT."
        )

        return None

    except requests.exceptions.RequestException as e:

        print(
            "TMDB REQUEST ERROR:",
            e
        )

        return None


  
         
      

        

               
      

# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def get_recommendations(
    title,
    number_of_movies=10
):

    if not title:

        return None

    title = title.strip()

    # EXACT MATCH
    exact_matches = movies[
        movies["title"].str.lower()
        == title.lower()
    ]

    if not exact_matches.empty:

        idx = exact_matches.index[0]

    else:

        # PARTIAL MATCH
        partial_matches = movies[
            movies["title"]
            .str.lower()
            .str.contains(
                title.lower(),
                regex=False,
                na=False
            )
        ]

        if partial_matches.empty:

            return None

        idx = partial_matches.index[0]

    # Get similarity scores
    similarity_row = cosine_sim.getrow(idx)

    sim_scores = list(
        zip(
            similarity_row.indices,
            similarity_row.data
        )
    )

    # Sort by similarity
    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove selected movie
    sim_scores = [
        item
        for item in sim_scores
        if item[0] != idx
    ]

    # Top N movies
    sim_scores = sim_scores[
        :number_of_movies
    ]

    recommendations = []

    for movie_index, similarity_score in sim_scores:

        movie_title = movies.iloc[
            movie_index
        ]["title"]

        poster = get_movie_poster(
            movie_title
        )

        recommendations.append({

            "title": movie_title,

            "poster": poster,

            "similarity": round(
                float(similarity_score) * 100,
                1
            )

        })

    return recommendations


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    selected_movie = ""

    error = ""

    if request.method == "POST":

        selected_movie = request.form.get(
            "movie",
            ""
        ).strip()

        if not selected_movie:

            error = "Please enter a movie name."

        else:

            recommendations = get_recommendations(
                selected_movie
            )

            if recommendations is None:

                error = (
                    "Movie not found. "
                    "Try entering part of "
                    "the movie title."
                )

                recommendations = []

    return render_template(
        "index.html",
        recommendations=recommendations,
        selected_movie=selected_movie,
        error=error
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False
    )