import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
# import math
import statistics
from dash import Dash, dcc, html, Input, Output, State, no_update

import nl_filter

# Read ANTHROPIC_API_KEY from the .env file next to this script. Variables
# already set in the shell take precedence, so exporting the key by hand
# overrides the file. A missing .env or a missing library is not an error. The
# app runs without a key and disables the natural-language box.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass



#################################################################################################################

JSON_PATH = "games.json"
# Parsing 675 MB of JSON takes minutes, so the cleaned dataframe is cached and
# rebuilt only when games.json changes. Delete this file to force a rebuild.
CACHE_PATH = "games_clean.pkl"

# The columns read by the app and the natural-language filter. The raw dataset
# also carries descriptions, screenshot URLs and trailers, which account for
# most of its size and are unused here.
KEEP_COLUMNS = [
    "appid", "name", "tags", "genres", "categories",
    "positive", "negative", "total_reviews", "review_score",
    "owners", "price_clean", "release_year", "genre_main",
    "median_playtime_forever", "average_playtime_forever",
    "windows", "mac", "linux",
]


def parse_owners(s):
    if not isinstance(s, str):
        return np.nan
    nums = re.findall(r"\d[\d,]*", s)
    if len(nums) == 0:
        return np.nan
    nums = [int(x.replace(",", "")) for x in nums]
    if len(nums) == 1:
        return nums[0]
    return (nums[0] + nums[1]) // 2


def get_first_genre(g):
    if isinstance(g, list) and len(g) > 0:
        return g[0]
    return "Unknown"


def extract_price(pkg_list, fallback):
    prices = []
    if isinstance(pkg_list, list):
        for pkg in pkg_list:
            subs = pkg.get("subs", [])
            if isinstance(subs, list):
                for sub in subs:
                    p = sub.get("price", None)
                    if isinstance(p, (int, float)):
                        prices.append(p)
    if prices:
        return min(prices)
    return fallback


NON_GAME_GENRES = {
    "Game Development",
    "Animation & Modeling",
    "Design & Illustration",
    "Education",
    "Video Production",
    "Photo Editing",
    "Web Publishing",
    "Software Training",
    "Utilities",
    "Audio Production"
}


def build_dataframe():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame.from_dict(raw, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "appid"}, inplace=True)

    df["owners"] = df["estimated_owners"].apply(parse_owners)

    df["total_reviews"] = df["positive"] + df["negative"]
    df = df[df["total_reviews"] > 0]
    df["review_score"] = df["positive"] / df["total_reviews"]

    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    df["genre_main"] = df["genres"].apply(get_first_genre)

    df = df[~df["genre_main"].isin(NON_GAME_GENRES)]

    df["price_clean"] = df.apply(lambda row: extract_price(row["packages"], row["price"]), axis=1)

    df = df[df["owners"].notna()]
    df = df[df["owners"] >= 100]
    df = df[df["positive"] >= 100]
    df = df[df["review_score"] >= 0.1]
    df = df[df["price_clean"] <= 150]

    return df[KEEP_COLUMNS]


def load_dataframe():
    source_time = os.path.getmtime(JSON_PATH)
    if os.path.exists(CACHE_PATH) and os.path.getmtime(CACHE_PATH) >= source_time:
        print(f"Loading cached dataframe from {CACHE_PATH}")
        return pd.read_pickle(CACHE_PATH)

    print(f"Building dataframe from {JSON_PATH} (this takes a few minutes)...")
    df = build_dataframe()
    df.to_pickle(CACHE_PATH)
    print(f"Cached {len(df)} games to {CACHE_PATH}")
    return df


df = load_dataframe()

#################################################################################################################

df["hidden_gem_score"] = df["review_score"] / np.sqrt(df["owners"])

df["hidden_gem_score_norm"] = (
    (df["hidden_gem_score"] - df["hidden_gem_score"].min()) /
    (df["hidden_gem_score"].max() - df["hidden_gem_score"].min())
)

# Lowercase tag, genre and category sets used for filtering, and the list of
# names the model is allowed to use when it builds a filter.
df = nl_filter.prepare_dataframe(df)
VOCAB = nl_filter.collect_vocabulary(df)

print(f"{len(df)} games loaded, {len(VOCAB['tags'])} tags in the filter vocabulary")

#################################################################################################################


app = Dash(__name__)

app.layout = html.Div([
    html.H1("Steam Hidden Gem Explorer"),

    html.Label("Describe what you're looking for"),
    html.Div([
        dcc.Input(
            id="nl-input",
            type="text",
            placeholder="cheap co-op games with good reviews that aren't too long",
            debounce=False,
            style={"width": "520px", "marginRight": "10px"}
        ),
        html.Button("Search", id="nl-search-button", n_clicks=0),
    ], style={"marginBottom": "8px"}),

    dcc.Loading(
        html.Div(id="nl-status", style={"minHeight": "40px", "marginBottom": "16px"}),
        type="default"
    ),

    html.Label("Or enter a tag directly (e.g. 'horror')"),
    dcc.Input(
        id="genre-input",
        type="text",
        placeholder="Type a tag...",
        style={"width": "300px", "margin-bottom": "20px"}
    ),

    dcc.Store(id="filter-spec"),

    dcc.Graph(id="scatter-plot"),
    dcc.Graph(id="tag-bar-chart"),
    dcc.Graph(id="year-bar-chart")
])

#################################################################################################################


@app.callback(
    [Output("filter-spec", "data"),
     Output("nl-status", "children")],
    [Input("nl-search-button", "n_clicks"),
     Input("nl-input", "n_submit")],
    State("nl-input", "value"),
    prevent_initial_call=True
)
def run_nl_search(n_clicks, n_submit, text):
    """Send the typed description to the API and store the returned filter spec."""
    if not (text or "").strip():
        return None, html.Div("Natural-language filter cleared.", style={"color": "#666"})

    try:
        spec = nl_filter.parse_query(text, VOCAB)
    except nl_filter.FilterError as e:
        # A failure leaves the existing view in place rather than blanking the
        # charts, and reports what happened.
        return no_update, html.Div(str(e), style={"color": "#b00020"})

    status = html.Div([
        html.Div(spec["interpretation"], style={"fontWeight": "bold"}),
        html.Div(nl_filter.describe_spec(spec), style={"color": "#555", "fontSize": "13px"}),
    ])
    return spec, status


@app.callback(
    [Output("scatter-plot", "figure"),
     Output("tag-bar-chart", "figure"),
     Output("year-bar-chart", "figure")],
    [Input("filter-spec", "data"),
     Input("genre-input", "value")]
)
def update_charts(spec, genre_filter):

    filtered = nl_filter.apply_filters(df, spec)

    if genre_filter is not None and genre_filter.strip() != "":
        g = genre_filter.strip().lower()

        def has_tag(tag_dict):
            if isinstance(tag_dict, dict):
                return any(g in tag.lower() for tag in tag_dict.keys())
            return False

        filtered = filtered[filtered["tags"].apply(has_tag)]


    scatter = go.Figure()

    scatter.add_trace(
        go.Scatter(
            x=filtered["owners"],
            y=filtered["review_score"],
            mode="markers",
            marker=dict(
                size=(filtered["price_clean"] + 1) ** 0.5 * 4,
                color=filtered["hidden_gem_score_norm"],
                colorscale="Viridis",
                opacity=0.65,
                showscale=True,
                colorbar=dict(title="Hidden Gem Score")
            ),
            text=filtered["name"],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Owners: %{x}<br>" +
                "Review Score: %{y:.2f}<br>" +
                "Price: $%{customdata[0]:.2f}<br>" +
                "Positive: %{customdata[1]}<br>" +
                "Negative: %{customdata[2]}<br>" +
                "Release Year: %{customdata[3]}<br>" +
                "<extra></extra>"
            ),
            customdata=np.stack([
                filtered["price_clean"],
                filtered["positive"],
                filtered["negative"],
                filtered["release_year"]
            ], axis=-1),
        )
    )

    scatter.update_xaxes(title="Estimated Owners (log scale)", type="log")
    scatter.update_yaxes(title="Review Score %")
    scatter.update_layout(title=f"Hidden Gem Scatter Plot ({len(filtered)} games)", height=650)


    tag_counts = {}

    for tags in filtered["tags"]:
        if isinstance(tags, dict):
            for tag in tags.keys():
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if tag_counts:
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        bar_tags = [t[0] for t in sorted_tags]
        bar_counts = [t[1] for t in sorted_tags]
    else:
        bar_tags = []
        bar_counts = []

    tag_bar = go.Figure(go.Bar(
        x=bar_counts,
        y=bar_tags,
        orientation="h"
    ))
    tag_bar.update_layout(
        title="Top Co-Occurring Tags",
        height=500,
        xaxis_title="Count"
    )



    year_counts = filtered["release_year"].value_counts().sort_index()

    year_bar = go.Figure(go.Bar(
        x=year_counts.index,
        y=year_counts.values
    ))
    year_bar.update_layout(
        title="Release Year Distribution",
        height=500,
        xaxis_title="Year",
        yaxis_title="Number of games released"
    )

    return scatter, tag_bar, year_bar



if __name__ == "__main__":
    app.run(debug=True)
