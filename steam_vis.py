import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
# import math
import statistics
from dash import Dash, dcc, html, Input, Output



#################################################################################################################

JSON_PATH = "games.json"  

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

df = pd.DataFrame.from_dict(raw, orient="index")
df.reset_index(inplace=True)
df.rename(columns={"index": "appid"}, inplace=True)





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

df["owners"] = df["estimated_owners"].apply(parse_owners)

df["total_reviews"] = df["positive"] + df["negative"]
df = df[df["total_reviews"] > 0]
df["review_score"] = df["positive"] / df["total_reviews"]

df["release_year"] = pd.to_datetime(
    df["release_date"], errors="coerce", infer_datetime_format=True
).dt.year

def get_first_genre(g):
    if isinstance(g, list) and len(g) > 0:
        return g[0]
    return "Unknown"

df["genre_main"] = df["genres"].apply(get_first_genre)



#################################################################################################################


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

df = df[~df["genre_main"].isin(NON_GAME_GENRES)]


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

df["price_clean"] = df.apply(lambda row: extract_price(row["packages"], row["price"]), axis=1)

#################################################################################################################

df = df[df["owners"].notna()]
df = df[df["owners"] >= 100]
df = df[df["positive"] >= 100]
df = df[df["review_score"] >= 0.1]
df = df[df["price_clean"] <= 150]

#################################################################################################################

df["hidden_gem_score"] = df["review_score"] / np.sqrt(df["owners"])





df["hidden_gem_score_norm"] = (
    (df["hidden_gem_score"] - df["hidden_gem_score"].min()) /
    (df["hidden_gem_score"].max() - df["hidden_gem_score"].min())
)

#################################################################################################################


app = Dash(__name__)

app.layout = html.Div([
    html.H1("Steam Hidden Gem Explorer"),

    html.Label("Enter a tag to filter (e.g. 'horror')"),
    dcc.Input(
        id="genre-input",
        type="text",
        placeholder="Type a tag...",
        style={"width": "300px", "margin-bottom": "20px"}
    ),

    dcc.Graph(id="scatter-plot"),
    dcc.Graph(id="tag-bar-chart"),
    dcc.Graph(id="year-bar-chart")
])

#################################################################################################################

@app.callback(
    [Output("scatter-plot", "figure"),
     Output("tag-bar-chart", "figure"),
     Output("year-bar-chart", "figure")],
    Input("genre-input", "value")
)
def update_charts(genre_filter):

    if genre_filter is None or genre_filter.strip() == "":
        filtered = df
    else:
        g = genre_filter.strip().lower()

        def has_tag(tag_dict):
            if isinstance(tag_dict, dict):
                return any(g in tag.lower() for tag in tag_dict.keys())
            return False

        filtered = df[df["tags"].apply(has_tag)]


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
    scatter.update_layout(title="Hidden Gem Scatter Plot", height=650)


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
