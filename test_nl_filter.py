"""Tests for the filtering logic.

These run without an API key, a network connection or the dataset, since the
model produces only a filter spec and the filtering itself is pandas applied to
the rows defined below.

    python test_nl_filter.py
"""

import pandas as pd

import nl_filter


# name, tags, genres, categories, price, review, owners, year, playtime(min), win, mac, linux
ROWS = [
    ("Cheap Coop Short",  {"Co-op": 9, "Indie": 5}, ["Indie"],  ["Multi-player", "Co-op"],  7.99, 0.92,   40000, 2021,  480, True,  True,  False),
    ("Cheap Coop Long",   {"Co-op": 9},             ["Action"], ["Multi-player", "Co-op"],  4.99, 0.88,   90000, 2019, 6000, True,  False, False),
    ("Pricey Coop Short", {"Co-op": 3},             ["Action"], ["Multi-player", "Co-op"], 49.99, 0.95, 2000000, 2023,  300, True,  False, True),
    ("Cheap Solo Short",  {"Horror": 7},            ["Indie"],  ["Single-player"],          3.99, 0.91,   12000, 2020,  200, True,  False, False),
    ("Cheap Coop NoData", {"Co-op": 2},             ["Indie"],  ["Co-op"],                  1.99, 0.99,     500, 2018,    0, True,  False, False),
    ("Free Coop Bad",     {"Co-op": 4},             ["Indie"],  ["Co-op"],                  0.00, 0.35,  800000, 2022,  900, True,  True,  True),
    # Added for the tag group tests. Medieval but not co-op.
    ("Medieval Solo",     {"Medieval": 8},          ["Indie"],  ["Single-player"],          9.99, 0.90,   30000, 2020,  600, True,  False, False),
    ("Medieval Coop",     {"Medieval": 8, "Co-op": 6}, ["Action"], ["Multi-player", "Co-op"], 14.99, 0.87, 60000, 2021, 1200, True, False, False),
]

COLUMNS = [
    "name", "tags", "genres", "categories", "price_clean", "review_score",
    "owners", "release_year", "median_playtime_forever", "windows", "mac", "linux",
]

df = nl_filter.prepare_dataframe(pd.DataFrame(ROWS, columns=COLUMNS))

results = []


def check(label, spec, expected):
    got = sorted(nl_filter.apply_filters(df, nl_filter.normalise_spec(spec))["name"].tolist())
    passed = got == sorted(expected)
    results.append(passed)
    print(f"{'ok  ' if passed else 'FAIL'} {label}")
    if not passed:
        print(f"       got={got}\n       exp={sorted(expected)}")


def check_value(label, passed, detail=""):
    results.append(passed)
    print(f"{'ok  ' if passed else 'FAIL'} {label}{(' -> ' + str(detail)) if detail else ''}")


# The query used in the README, "cheap co-op games with good reviews that
# aren't too long".
check("cheap co-op, good reviews, not too long",
      {"category_groups": [["Co-op"]], "price_max": 10,
       "review_score_min": 0.8, "playtime_max_hours": 15},
      ["Cheap Coop Short"])

# "medieval games to play with friends" exposed the earlier flat tag list. Two
# concepts, so two groups, combined with AND.
check("two concepts are ANDed across groups",
      {"tag_groups": [["Medieval"]], "category_groups": [["Multi-player", "Co-op"]]},
      ["Medieval Coop"])

# The same names in one group form an OR, which was the earlier behaviour. It
# matches every co-op game whether or not it is medieval.
check("one group is an OR (the old, wrong behaviour)",
      {"tag_groups": [["Medieval", "Co-op"]]},
      ["Medieval Solo", "Medieval Coop", "Cheap Coop Short", "Cheap Coop Long",
       "Pricey Coop Short", "Cheap Coop NoData", "Free Coop Bad"])

# Alternatives for one concept belong in the same group.
check("alternatives within a group are OR'd",
      {"tag_groups": [["Horror", "Medieval"]], "review_score_min": 0.9},
      ["Cheap Solo Short", "Medieval Solo"])

# A playtime of 0 means no data rather than a very short game. Without this
# every unplayed game matches a search for short games.
check("playtime filter excludes zero-playtime rows",
      {"playtime_max_hours": 15},
      ["Cheap Coop Short", "Pricey Coop Short", "Cheap Solo Short", "Free Coop Bad",
       "Medieval Solo"])

check("free_only", {"free_only": True}, ["Free Coop Bad"])

check("separate groups require every concept",
      {"tag_groups": [["Co-op"], ["Indie"]]}, ["Cheap Coop Short"])

check("a single group requires only one name",
      {"tag_groups": [["Horror", "Indie"]]}, ["Cheap Coop Short", "Cheap Solo Short"])

check("platforms are ANDed", {"platforms": ["mac", "linux"]}, ["Free Coop Bad"])

check("owners_max finds the obscure ones", {"owners_max": 50000},
      ["Cheap Coop Short", "Cheap Solo Short", "Cheap Coop NoData", "Medieval Solo"])

check("year range", {"year_min": 2021, "year_max": 2023},
      ["Cheap Coop Short", "Pricey Coop Short", "Free Coop Bad", "Medieval Coop"])

check("sort + limit",
      {"sort_by": "review_score", "sort_desc": True, "limit": 2},
      ["Cheap Coop NoData", "Pricey Coop Short"])

check("empty spec is a no-op", {}, [r[0] for r in ROWS])

# A spec of None means no search has been made yet.
check_value("None spec is a no-op",
            sorted(nl_filter.apply_filters(df, None)["name"].tolist())
            == sorted(r[0] for r in ROWS))

# The model should return a value from 0.0 to 1.0. A percentage is rescaled
# rather than clamped to 1.0, which would filter out every row.
rescaled = nl_filter.normalise_spec({"review_score_min": 85})["review_score_min"]
check_value("percentage review score is rescaled", abs(rescaled - 0.85) < 1e-9, rescaled)

# Values outside the schema enum are dropped rather than passed to pandas.
platforms = nl_filter.normalise_spec({"platforms": ["windows", "steamdeck"]})["platforms"]
check_value("unknown platform is dropped", platforms == ["windows"], platforms)

# A flat list where groups are expected does not become a filter that matches
# nothing.
check_value("a stray flat list is coerced, not silently emptied",
            nl_filter.normalise_spec({"tag_groups": ["Medieval"]})["tag_groups"] == [["Medieval"]])

check_value("empty groups are dropped",
            nl_filter.normalise_spec({"tag_groups": [[], [""], ["Medieval"]]})["tag_groups"]
            == [["Medieval"]])

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
