"""
Natural-language filtering for the Steam Hidden Gem Explorer.

Turns a typed description such as "cheap co-op games with good reviews that
aren't too long" into a filter spec, using one Claude API call with structured
outputs so the reply always matches FILTER_SCHEMA. The spec is then applied to
the dataframe with pandas.

build_system_prompt(vocab) describes the available fields and valid values.
parse_query(text, vocab) returns a filter spec and raises FilterError.
apply_filters(df, spec) returns the filtered dataframe.

The API call produces parameters only. The filtering itself is pandas code.
"""

import json
import os

import anthropic
import pandas as pd


# Haiku is the cheapest model and the schema does most of the structural work,
# so the model only maps wording onto thresholds. Switching to "claude-opus-5"
# also requires restoring the effort setting in parse_query, which Haiku
# rejects.
MODEL = "claude-haiku-4-5"

# How many of the most common tags to list in the prompt. The dataset holds
# thousands of distinct tags and the long tail costs tokens without adding
# useful vocabulary.
TAG_VOCAB_SIZE = 200


class FilterError(Exception):
    """Raised when a query could not be turned into a usable filter spec."""


# ---------------------------------------------------------------------------
# What the model is allowed to return
# ---------------------------------------------------------------------------

def _nullable(inner, description):
    """A field that is either `inner` or null.

    Written as anyOf rather than {"type": ["number", "null"]} because only
    anyOf is documented as supported by structured outputs.
    """
    return {"anyOf": [inner, {"type": "null"}], "description": description}


# Every field is listed as required and is nullable. Structured outputs needs
# all fields present, and it also makes the model state a value for each filter
# rather than omitting the ones it is unsure about.
FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        # Groups rather than flat lists. One group holds one concept from the
        # request. A game matches a group by having any name in it and must
        # match every group. A flat OR list cannot express "medieval AND
        # multiplayer", and a flat AND list requires every tag at once.
        "tag_groups": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
            "description": (
                "One group per concept the person asked for. A game must match "
                "every group, and matches a group by having any tag in it. "
                'Example: "medieval roguelikes" -> [["Medieval"], ["Roguelike", "Rogue-lite"]].'
            ),
        },
        # There is no genre field. Every store genre worth searching for
        # (Action, RPG, Indie, Gore, Racing) also exists as a tag, and the
        # genre-only values are non-game software that steam_vis already
        # removes. When genres were offered as a separate filter the model
        # added the Gore genre to horror queries unprompted, which reduced 23
        # results to 0.
        "category_groups": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
            "description": (
                "Same grouping rule, over Steam feature categories. This is "
                "where multiplayer/co-op/controller support live."
            ),
        },
        "price_min": _nullable({"type": "number"}, "Minimum price in USD."),
        "price_max": _nullable({"type": "number"}, "Maximum price in USD."),
        "free_only": {"type": "boolean", "description": "Keep only games priced at 0."},
        "review_score_min": _nullable(
            {"type": "number"}, "Minimum share of positive reviews, 0.0 to 1.0."
        ),
        "review_score_max": _nullable(
            {"type": "number"}, "Maximum share of positive reviews, 0.0 to 1.0."
        ),
        "owners_min": _nullable({"type": "integer"}, "Minimum estimated owners."),
        "owners_max": _nullable({"type": "integer"}, "Maximum estimated owners."),
        "year_min": _nullable({"type": "integer"}, "Earliest release year."),
        "year_max": _nullable({"type": "integer"}, "Latest release year."),
        "playtime_min_hours": _nullable(
            {"type": "number"}, "Minimum median playtime in hours."
        ),
        "playtime_max_hours": _nullable(
            {"type": "number"}, "Maximum median playtime in hours."
        ),
        "platforms": {
            "type": "array",
            "items": {"type": "string", "enum": ["windows", "mac", "linux"]},
            "description": "Keep only games running on all of these platforms.",
        },
        "sort_by": _nullable(
            {
                "type": "string",
                "enum": [
                    "hidden_gem_score_norm",
                    "review_score",
                    "owners",
                    "price_clean",
                    "release_year",
                ],
            },
            "Column to sort by before applying 'limit'.",
        ),
        "sort_desc": {"type": "boolean", "description": "Sort descending rather than ascending."},
        "limit": _nullable({"type": "integer"}, "Keep only the first N games after sorting."),
        "interpretation": {
            "type": "string",
            "description": (
                "One plain sentence, shown to the user, saying how the request "
                "was read and which thresholds were chosen."
            ),
        },
    },
    "required": [
        "tag_groups",
        "category_groups",
        "price_min",
        "price_max",
        "free_only",
        "review_score_min",
        "review_score_max",
        "owners_min",
        "owners_max",
        "year_min",
        "year_max",
        "playtime_min_hours",
        "playtime_max_hours",
        "platforms",
        "sort_by",
        "sort_desc",
        "limit",
        "interpretation",
    ],
    "additionalProperties": False,
}


EMPTY_SPEC = {
    "tag_groups": [],
    "category_groups": [],
    "price_min": None,
    "price_max": None,
    "free_only": False,
    "review_score_min": None,
    "review_score_max": None,
    "owners_min": None,
    "owners_max": None,
    "year_min": None,
    "year_max": None,
    "playtime_min_hours": None,
    "playtime_max_hours": None,
    "platforms": [],
    "sort_by": None,
    "sort_desc": True,
    "limit": None,
    "interpretation": "",
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_system_prompt(vocab):
    """Describe the dataset to the model.

    vocab comes from collect_vocabulary() and holds the tag and category names
    present in the data. Without that list the model returns plausible tag
    names that exist in no rows.
    """
    return f"""You translate a person's description of the Steam games they want into filter parameters for a dataset of Steam games.

Each game in the dataset has:
- tags: community tags, a game usually has 5-20 of them
- categories: Steam feature categories (this is where single-player, multiplayer, co-op and controller support live, NOT in tags)
- price_clean: price in USD, 0 for free games
- review_score: share of reviews that are positive, from 0.0 to 1.0
- owners: estimated number of owners
- release_year: year of release
- median_playtime_forever: median playtime, which the filters express in hours

Valid tags (use these exact strings, nothing else):
{", ".join(vocab["tags"])}

Valid categories (use these exact strings, nothing else, there is no "Local Co-op"):
{", ".join(vocab["categories"])}

Grouping. tag_groups and category_groups are lists of groups, not flat lists. One group is one concept from the request. A game must match EVERY group, and it matches a group by having ANY name in that group. So put alternative ways of expressing the SAME concept together in one group, and separate concepts in separate groups.

  "medieval games to play with friends"
    tag_groups:      [["Medieval"]]
    category_groups: [["Multi-player", "Co-op", "Online Co-op", "LAN Co-op"]]
    -> medieval AND (any of those multiplayer categories)

  "scary roguelikes"
    tag_groups: [["Horror", "Psychological Horror"], ["Roguelike", "Rogue-lite"]]
    -> (horror or psychological horror) AND (roguelike or rogue-lite)

Putting two different concepts in one group is the most damaging mistake you can make here: ["Medieval", "Co-op"] matches every co-op game ever made, most of which are not medieval.

How to read vague wording. These are starting points, adjust them when the request implies something stronger or weaker:
- "cheap" -> price_max 10; "very cheap" or "budget" -> price_max 5; "free" -> free_only true
- "good reviews" -> review_score_min 0.8; "great" or "amazing reviews" -> 0.9; "decent" -> 0.7
- "hidden gem", "underrated", "obscure", "nobody plays" -> owners_max 200000
- "popular" or "well known" -> owners_min 1000000
- "short" or "not too long" -> playtime_max_hours 15; "very short" -> 5; "long" or "meaty" -> playtime_min_hours 40
- "recent" or "new" -> year_min 2022; "old" or "classic" -> year_max 2010
- "co-op" -> the Co-op category; "multiplayer" -> the Multi-player category

Rules:
- One group per concept the person actually named. Every group narrows the results, so a group they did not ask for silently throws away games they wanted. Asking for horror games does not imply gore; asking for co-op games does not imply Online PvP.
- Leave a field null, or a list empty, when the request does not imply it.
- Express one concept in one place. If "co-op" is already a category group, do not also add it as a tag group - that filters twice for the same idea and drops games that are tagged one way but not the other.
- Prefer a category over a tag when the concept is one of the categories, since categories are more reliable.
- Only set sort_by and limit if the person asked for a ranking or a specific number of results.
- Write interpretation as one sentence naming the thresholds you picked, so the person can see whether you read them correctly."""


def collect_vocabulary(df, tag_limit=TAG_VOCAB_SIZE):
    """Return the tag, genre and category names present in the data."""
    tag_counts = {}
    for tags in df["tags"]:
        if isinstance(tags, dict):
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    top_tags = sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:tag_limit]

    genres = set()
    for g in df["genres"]:
        if isinstance(g, list):
            genres.update(g)

    categories = {}
    for c in df["categories"]:
        if isinstance(c, list):
            for cat in c:
                categories[cat] = categories.get(cat, 0) + 1

    return {
        "tags": [t for t, _ in top_tags],
        "genres": sorted(genres),
        # Categories are a short fixed list. Ordering by frequency puts
        # Single-player, Multi-player and Co-op first.
        "categories": [c for c, _ in sorted(categories.items(), key=lambda kv: kv[1], reverse=True)],
    }


# ---------------------------------------------------------------------------
# The API call
# ---------------------------------------------------------------------------

_client = None

# The value shipped in .env.example. It is checked by name because it is a
# non-empty string and would otherwise pass the missing-key check and fail
# later as a rejected key.
PLACEHOLDER_KEY = "sk-ant-paste-your-key-here"


def get_client():
    """Build the Anthropic client on first use.

    Deferred so that importing this module does not fail when no key is set.
    """
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise FilterError(
                "No API key found. Put your key in the .env file next to "
                "steam_vis.py and restart the app. The tag box below still "
                "works without it."
            )
        if key == PLACEHOLDER_KEY or "paste-your-key" in key:
            raise FilterError(
                "The .env file still has the placeholder in it. Replace it with "
                "a real key from console.anthropic.com/settings/keys, then "
                "restart the app."
            )
        _client = anthropic.Anthropic(api_key=key)
    return _client


def parse_query(text, vocab):
    """Turn a sentence into a filter spec.

    Raises FilterError with a message suitable for display. Callers should
    treat FilterError as a signal to keep the current view and show the
    message.
    """
    text = (text or "").strip()
    if not text:
        raise FilterError("Type what you're looking for, then press Search.")

    client = get_client()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            # The system prompt is identical on every request, so it is marked
            # for caching. On Haiku 4.5 this has no effect, since the prompt is
            # roughly 1300 tokens and Haiku caches prefixes of 4096 or more.
            # The marker is kept because it applies once the vocabulary grows
            # or the model changes.
            system=[
                {
                    "type": "text",
                    "text": build_system_prompt(vocab),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            # No effort setting here, since Haiku 4.5 rejects it. Restore
            # the effort setting when switching to an Opus or Sonnet model.
            output_config={"format": {"type": "json_schema", "schema": FILTER_SCHEMA}},
            messages=[{"role": "user", "content": text}],
        )
    except anthropic.AuthenticationError:
        raise FilterError("The API key was rejected. Check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise FilterError("Rate limited by the API. Wait a few seconds and try again.")
    except anthropic.APIConnectionError:
        raise FilterError("Could not reach the API. Check your internet connection.")
    except anthropic.APIStatusError as e:
        raise FilterError(f"API error {e.status_code}: {e.message}")

    if response.stop_reason == "refusal":
        raise FilterError("The model declined to answer that request. Try rephrasing it.")

    reply = next((b.text for b in response.content if b.type == "text"), None)
    if reply is None:
        raise FilterError("The model returned no text. Try again.")

    # output_config.format guarantees valid JSON matching the schema. Parsed
    # defensively anyway so that no malformed reply reaches the plotting code.
    try:
        spec = json.loads(reply)
    except json.JSONDecodeError:
        raise FilterError("The model's reply was not valid JSON. Try rephrasing your query.")

    if not isinstance(spec, dict):
        raise FilterError("The model's reply was not a filter spec. Try rephrasing your query.")

    return normalise_spec(spec, vocab)


def normalise_spec(spec, vocab=None):
    """Fill in missing fields and clamp values into usable ranges.

    When vocab is supplied, tag and category names that do not occur in the
    data are removed. The model occasionally returns a plausible name such as
    "Local Co-op" where the real category is "Shared/Split Screen Co-op". Such
    a name matches no rows, so inside a group of alternatives it does nothing
    visible, and as the only name in a group it empties the results. Removed
    names are recorded under _dropped_names so the eval suite can still count
    how often this happens.
    """
    clean = dict(EMPTY_SPEC)

    for key in EMPTY_SPEC:
        if key in spec and spec[key] is not None:
            clean[key] = spec[key]

    # The model sometimes returns a percentage rather than a fraction.
    for key in ("review_score_min", "review_score_max"):
        value = clean[key]
        if value is None:
            continue
        value = float(value)
        if value > 1.0:
            value = value / 100.0
        clean[key] = min(max(value, 0.0), 1.0)

    if clean["limit"] is not None:
        clean["limit"] = max(1, int(clean["limit"]))

    clean["platforms"] = [p for p in clean["platforms"] if p in ("windows", "mac", "linux")]

    allowed = None
    if vocab:
        allowed = {
            "tag_groups": {t.lower() for t in vocab.get("tags", [])},
            "category_groups": {c.lower() for c in vocab.get("categories", [])},
        }

    # Accept a flat list where a list of groups is expected. The schema should
    # prevent this, but an empty or malformed group would become a filter that
    # matches nothing and would empty the charts without explanation.
    dropped = []
    for key in ("tag_groups", "category_groups"):
        groups = []
        for group in clean[key] or []:
            if isinstance(group, str):
                group = [group]
            names = [n for n in group if isinstance(n, str) and n.strip()]
            if allowed is not None:
                kept = [n for n in names if n.lower() in allowed[key]]
                dropped.extend(n for n in names if n not in kept)
                names = kept
            if names:
                groups.append(names)
        clean[key] = groups

    clean["_dropped_names"] = dropped

    return clean


# ---------------------------------------------------------------------------
# Applying the spec
# ---------------------------------------------------------------------------

def prepare_dataframe(df):
    """Add lowercase lookup columns used by apply_filters.

    Called once, after the dataframe is built, so the sets are not rebuilt on
    every query.
    """
    df = df.copy()
    df["_tags_lc"] = df["tags"].apply(
        lambda t: {tag.lower() for tag in t} if isinstance(t, dict) else set()
    )
    df["_genres_lc"] = df["genres"].apply(
        lambda g: {x.lower() for x in g} if isinstance(g, list) else set()
    )
    df["_categories_lc"] = df["categories"].apply(
        lambda c: {x.lower() for x in c} if isinstance(c, list) else set()
    )
    return df


def apply_filters(df, spec):
    """Apply a filter spec to the dataframe using pandas only."""
    if not spec:
        return df

    mask = pd.Series(True, index=df.index)

    # OR within a group and AND across groups. A game must satisfy every group
    # and satisfies one by carrying any single name in it.
    for key, column in (("tag_groups", "_tags_lc"),
                        ("category_groups", "_categories_lc")):
        for group in spec.get(key) or []:
            wanted = {name.lower() for name in group}
            if wanted:
                mask &= df[column].apply(lambda have, w=wanted: bool(have & w))

    if spec.get("free_only"):
        mask &= df["price_clean"] == 0
    if spec.get("price_min") is not None:
        mask &= df["price_clean"] >= spec["price_min"]
    if spec.get("price_max") is not None:
        mask &= df["price_clean"] <= spec["price_max"]

    if spec.get("review_score_min") is not None:
        mask &= df["review_score"] >= spec["review_score_min"]
    if spec.get("review_score_max") is not None:
        mask &= df["review_score"] <= spec["review_score_max"]

    if spec.get("owners_min") is not None:
        mask &= df["owners"] >= spec["owners_min"]
    if spec.get("owners_max") is not None:
        mask &= df["owners"] <= spec["owners_max"]

    if spec.get("year_min") is not None:
        mask &= df["release_year"] >= spec["year_min"]
    if spec.get("year_max") is not None:
        mask &= df["release_year"] <= spec["year_max"]

    # A playtime of 0 means no data rather than a very short game, so any
    # playtime filter excludes those rows. Without this every unplayed game
    # matches a search for short games.
    playtime_min = spec.get("playtime_min_hours")
    playtime_max = spec.get("playtime_max_hours")
    if playtime_min is not None or playtime_max is not None:
        playtime_hours = df["median_playtime_forever"] / 60.0
        mask &= df["median_playtime_forever"] > 0
        if playtime_min is not None:
            mask &= playtime_hours >= playtime_min
        if playtime_max is not None:
            mask &= playtime_hours <= playtime_max

    for platform in spec.get("platforms") or []:
        mask &= df[platform] == True  # noqa: E712 - pandas column comparison

    filtered = df[mask]

    if spec.get("sort_by") and spec["sort_by"] in filtered.columns:
        filtered = filtered.sort_values(spec["sort_by"], ascending=not spec.get("sort_desc", True))

    if spec.get("limit"):
        filtered = filtered.head(spec["limit"])

    return filtered


def describe_spec(spec):
    """Return a readable summary of the active filters."""
    if not spec:
        return ""

    parts = []
    # Groups are rendered in brackets so the AND and OR structure is visible,
    # for example "tags [Medieval] and [Roguelike or Rogue-lite]".
    for key, label in (("tag_groups", "tags"),
                       ("category_groups", "categories")):
        groups = spec.get(key) or []
        if groups:
            rendered = " and ".join("[" + " or ".join(g) + "]" for g in groups)
            parts.append(f"{label}: {rendered}")
    if spec["free_only"]:
        parts.append("free only")
    if spec["price_min"] is not None:
        parts.append(f"price >= ${spec['price_min']:g}")
    if spec["price_max"] is not None:
        parts.append(f"price <= ${spec['price_max']:g}")
    if spec["review_score_min"] is not None:
        parts.append(f"review score >= {spec['review_score_min']:.0%}")
    if spec["review_score_max"] is not None:
        parts.append(f"review score <= {spec['review_score_max']:.0%}")
    if spec["owners_min"] is not None:
        parts.append(f"owners >= {spec['owners_min']:,}")
    if spec["owners_max"] is not None:
        parts.append(f"owners <= {spec['owners_max']:,}")
    if spec["year_min"] is not None:
        parts.append(f"released {spec['year_min']} or later")
    if spec["year_max"] is not None:
        parts.append(f"released {spec['year_max']} or earlier")
    if spec["playtime_min_hours"] is not None:
        parts.append(f"playtime >= {spec['playtime_min_hours']:g}h")
    if spec["playtime_max_hours"] is not None:
        parts.append(f"playtime <= {spec['playtime_max_hours']:g}h")
    if spec["platforms"]:
        parts.append("runs on " + ", ".join(spec["platforms"]))
    if spec["sort_by"]:
        parts.append(f"sorted by {spec['sort_by']}" + (" (desc)" if spec["sort_desc"] else " (asc)"))
    if spec["limit"]:
        parts.append(f"top {spec['limit']}")

    return " | ".join(parts) if parts else "no filters"
