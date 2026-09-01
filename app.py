from flask import Flask, render_template, request, send_from_directory
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


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(
    __name__,
    static_folder="public",
    static_url_path=""
)

from flask import send_from_directory


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

@app.route("/style.css")
def style():
    return send_from_directory(
        os.path.join(BASE_DIR, "public"),
        "style.css"
    )

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


# Convert genres to string

movies["genres"] = (
    movies["genres"]
    .astype(str)
)


# MovieLens format:
# Action|Adventure|Sci-Fi
#
# Convert to:
# action adventure sci-fi

movies["genres"] = (
    movies["genres"]
    .str.replace(
        "|",
        " ",
        regex=False
    )
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
        print("TMDB_API_KEY is not configured.")
        return None

    try:

        # --------------------------------------
        # STEP 1: Clean the movie title
        # --------------------------------------

        clean_title = title.strip()

        # Example:
        # "Toy Story (1995)"
        # becomes:
        # "Toy Story"

        import re

        title_without_year = re.sub(
            r"\s*\(\d{4}\)\s*$",
            "",
            clean_title
        ).strip()


        # --------------------------------------
        # STEP 2: Extract release year
        # --------------------------------------

        year_match = re.search(
            r"\((\d{4})\)\s*$",
            clean_title
        )

        year = None

        if year_match:
            year = year_match.group(1)


        # --------------------------------------
        # STEP 3: Search TMDB
        # --------------------------------------

        url = (
            "https://api.themoviedb.org/3/"
            "search/movie"
        )


        params = {
            "api_key": TMDB_API_KEY,
            "query": title_without_year,
            "language": "en-US",
            "include_adult": False
        }


        # If year exists, tell TMDB the year
        if year:
            params["year"] = year


        response = requests.get(
            url,
            params=params,
            timeout=5
        )


        if response.status_code != 200:

            print(
                "TMDB API error:",
                response.status_code
            )

            return None


        data = response.json()

        results = data.get(
            "results",
            []
        )


        # --------------------------------------
        # STEP 4: If no result, search again
        # without the year
        # --------------------------------------

        if not results:

            params = {
                "api_key": TMDB_API_KEY,
                "query": title_without_year,
                "language": "en-US",
                "include_adult": False
            }

            response = requests.get(
                url,
                params=params,
                timeout=5
            )

            if response.status_code != 200:
                return None

            data = response.json()

            results = data.get(
                "results",
                []
            )


        # --------------------------------------
        # STEP 5: Check results
        # --------------------------------------

        if not results:

            print(
                "No TMDB result for:",
                title
            )

            return None


        # --------------------------------------
        # STEP 6: Find first movie
        # with a poster
        # --------------------------------------

        for movie in results:

            poster_path = movie.get(
                "poster_path"
            )

            if poster_path:

                poster_url = (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

                return poster_url


        # No poster found

        print(
            "Movie found but poster unavailable:",
            title
        )

        return None


    except requests.RequestException as e:

        print(
            "TMDB connection error:",
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


    # --------------------------------------
    # EXACT MATCH
    # --------------------------------------

    exact_matches = movies[
        movies["title"].str.lower()
        == title.lower()
    ]


    if not exact_matches.empty:

        idx = exact_matches.index[0]


    else:

        # ----------------------------------
        # PARTIAL MATCH
        # ----------------------------------

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


    # --------------------------------------
    # GET SIMILARITY SCORES
    # --------------------------------------

    similarity_row = cosine_sim.getrow(
        idx
    )


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


    # Get top N

    sim_scores = sim_scores[
        :number_of_movies
    ]


    # --------------------------------------
    # CREATE RECOMMENDATION DATA
    # --------------------------------------

    recommendations = []


    for movie_index, similarity_score in sim_scores:

        movie_title = movies.iloc[
            movie_index
        ]["title"]


        # Get TMDB poster

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

@app.route(
    "/",
    methods=["GET", "POST"]
)

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

            error = (
                "Please enter a movie name."
            )


        else:

            recommendations = (
                get_recommendations(
                    selected_movie
                )
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
        debug=True
    )