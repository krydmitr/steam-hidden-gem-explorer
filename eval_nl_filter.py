"""Evaluation suite for the natural-language filter.

test_nl_filter.py checks the filtering code. This file checks the model. It
sends a fixed set of queries to the API, applies the returned specs to the real
dataset and asserts properties of both the spec and the resulting games.

Exact specs cannot be asserted, since several different specs are correct for
the same query. The assertions are therefore properties. A query naming a price
must set a price bound, a query naming one concept must not add a second, and
the games that come back must actually carry the tag that was asked for.

    python eval_nl_filter.py                 one run of every case
    python eval_nl_filter.py --runs 3        three runs, to see variance
    python eval_nl_filter.py --only medieval run the cases matching a substring
    python eval_nl_filter.py --no-save       do not record the run

Each run costs roughly one API call per case per run, a fraction of a cent
each. Results are written to eval_results/ and compared against the previous
run so that a prompt change shows up as a named regression rather than a
changed score.
"""

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import nl_filter


CACHE_PATH = "games_clean.pkl"
RESULTS_DIR = pathlib.Path("eval_results")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
#
# Each case is a query plus the properties its answer must have.
#
#   sets         fields that must hold a value
#   unset        fields that must be empty or null
#   bounds       field -> (low, high), an inclusive range for a numeric field
#   min_results  the filtered dataframe must hold at least this many games
#   max_results  and at most this many
#   tag_share    (tag, fraction), the share of returned games carrying that tag
#   groups       field -> number of separate groups expected
#
# Every case is additionally checked for hallucinated tag and category names
# and for a non-empty interpretation string.

CASES = [
    {
        "name": "readme_headline",
        "query": "cheap co-op games with good reviews that aren't too long",
        "sets": ["price_max", "category_groups", "review_score_min", "playtime_max_hours"],
        "bounds": {"price_max": (1, 20), "review_score_min": (0.6, 0.95),
                   "playtime_max_hours": (2, 30)},
        "min_results": 1,
    },
    {
        # The query that exposed the flat tag list. Medieval and the
        # multiplayer categories are separate concepts, so the games returned
        # must actually be medieval.
        "name": "two_concepts",
        "query": "medieval games to play with friends",
        "sets": ["tag_groups", "category_groups"],
        "tag_share": ("medieval", 0.9),
        "min_results": 1,
    },
    {
        # This returned 0 games while the model was adding the Gore genre to
        # horror queries.
        "name": "no_unrequested_narrowing",
        "query": "obscure horror games from before 2015 with great reviews",
        "sets": ["tag_groups", "owners_max", "year_max", "review_score_min"],
        "min_results": 1,
    },
    {
        "name": "bare_concept_adds_nothing",
        "query": "horror games",
        "sets": ["tag_groups"],
        "unset": ["price_max", "price_min", "review_score_min", "owners_max",
                  "owners_min", "year_min", "year_max", "playtime_max_hours",
                  "playtime_min_hours", "limit", "free_only"],
        "min_results": 500,
    },
    {
        "name": "free_and_platform",
        "query": "free multiplayer games that run on linux",
        "sets": ["free_only", "platforms", "category_groups"],
        "min_results": 1,
    },
    {
        "name": "ranking_and_limit",
        "query": "the 20 best-reviewed indie games nobody owns",
        "sets": ["limit", "sort_by", "owners_max"],
        "bounds": {"limit": (20, 20)},
        "max_results": 20,
        "min_results": 1,
    },
    {
        "name": "price_explicit",
        "query": "games under 5 dollars",
        "sets": ["price_max"],
        "bounds": {"price_max": (5, 5)},
        "unset": ["review_score_min", "owners_max", "playtime_max_hours"],
        "min_results": 1,
    },
    {
        "name": "review_explicit",
        "query": "games with at least 90 percent positive reviews",
        "sets": ["review_score_min"],
        "bounds": {"review_score_min": (0.88, 0.92)},
        "unset": ["price_max", "owners_max"],
        "min_results": 1,
    },
    {
        "name": "very_cheap_is_lower",
        "query": "very cheap indie games",
        "sets": ["price_max"],
        "bounds": {"price_max": (1, 6)},
        "min_results": 1,
    },
    {
        # Direction matters. "long" must set a minimum, not a maximum.
        "name": "long_is_a_minimum",
        "query": "long meaty rpgs to sink time into",
        "sets": ["playtime_min_hours"],
        "unset": ["playtime_max_hours"],
        "min_results": 1,
    },
    {
        # The negated form must set the opposite bound.
        "name": "negation_is_a_maximum",
        "query": "games that aren't too long",
        "sets": ["playtime_max_hours"],
        "unset": ["playtime_min_hours"],
        "min_results": 1,
    },
    {
        "name": "popular_is_a_floor",
        "query": "popular well known games",
        "sets": ["owners_min"],
        "unset": ["owners_max"],
        "min_results": 1,
    },
    {
        "name": "obscure_is_a_ceiling",
        "query": "hidden gems nobody has heard of",
        "sets": ["owners_max"],
        "unset": ["owners_min"],
        "min_results": 1,
    },
    {
        "name": "recent",
        "query": "recent indie games",
        "sets": ["year_min"],
        "unset": ["year_max"],
        "min_results": 1,
    },
    {
        "name": "two_tag_concepts",
        "query": "relaxing puzzle games",
        "sets": ["tag_groups"],
        "groups": {"tag_groups": 2},
        "min_results": 1,
    },
    {
        "name": "category_pair",
        "query": "co-op games with full controller support",
        "sets": ["category_groups"],
        "groups": {"category_groups": 2},
        "min_results": 1,
    },
    {
        # Asserted on the games returned rather than on which field was used.
        # Answering with the Singleplayer tag instead of the Single-player
        # category is equally correct, and an earlier version of this case
        # failed it for choosing the tag.
        "name": "single_player",
        "query": "single player story driven games",
        "category_share": ("single-player", 0.9),
        "min_results": 1,
    },
    {
        # Nonsense input must not crash and must not invent filters.
        "name": "nonsense_input",
        "query": "asdfghjkl qwertyuiop",
        "unset": ["price_max", "review_score_min", "owners_max", "year_min",
                  "year_max", "playtime_max_hours", "limit"],
        "allow_empty": True,
    },
]


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def is_set(spec, field):
    value = spec.get(field)
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, str)):
        return len(value) > 0
    return value is not None


def check_case(case, spec, df, vocab):
    """Return a list of failure strings. Empty means the case passed."""
    failures = []

    for field in case.get("sets", []):
        if not is_set(spec, field):
            failures.append(f"{field} was not set")

    for field in case.get("unset", []):
        if is_set(spec, field):
            failures.append(f"{field} was set to {spec.get(field)!r} but was not asked for")

    for field, (low, high) in case.get("bounds", {}).items():
        value = spec.get(field)
        if value is None:
            continue
        if not (low <= value <= high):
            failures.append(f"{field} was {value}, outside {low} to {high}")

    for field, expected in case.get("groups", {}).items():
        actual = len(spec.get(field) or [])
        if actual < expected:
            failures.append(
                f"{field} held {actual} group(s), expected at least {expected} "
                f"so the concepts combine with AND"
            )

    # normalise_spec removes names that do not occur in the data and records
    # them, so the app is protected while the eval can still count how often
    # the model invents one.
    for name in spec.get("_dropped_names") or []:
        failures.append(f"invented the name {name!r}, which is not in the data")

    if not (spec.get("interpretation") or "").strip():
        failures.append("interpretation was empty")

    got = nl_filter.apply_filters(df, spec)
    count = len(got)

    if "min_results" in case and count < case["min_results"]:
        failures.append(f"returned {count} games, expected at least {case['min_results']}")
    if "max_results" in case and count > case["max_results"]:
        failures.append(f"returned {count} games, expected at most {case['max_results']}")

    for key, column, label in (("tag_share", "_tags_lc", "tag"),
                               ("category_share", "_categories_lc", "category")):
        if key in case and count:
            name, minimum = case[key]
            share = got[column].apply(lambda have, n=name: n in have).mean()
            if share < minimum:
                failures.append(
                    f"only {share:.0%} of the {count} games carry the {name!r} {label}, "
                    f"expected at least {minimum:.0%}"
                )

    return failures, count


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_once(case, df, vocab):
    started = time.time()
    try:
        spec = nl_filter.parse_query(case["query"], vocab)
    except nl_filter.FilterError as e:
        return {"passed": False, "failures": [f"API call failed. {e}"],
                "results": 0, "seconds": time.time() - started, "spec": None}

    failures, count = check_case(case, spec, df, vocab)
    return {
        "passed": not failures,
        "failures": failures,
        "results": count,
        "seconds": time.time() - started,
        "spec": {k: v for k, v in spec.items() if k != "interpretation"},
    }


def load_previous():
    """The most recent saved run, or None."""
    if not RESULTS_DIR.exists():
        return None
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1,
                        help="how many times to run each case")
    parser.add_argument("--only", default="",
                        help="run only cases whose name or query contains this")
    parser.add_argument("--no-save", action="store_true",
                        help="do not write the run to eval_results/")
    args = parser.parse_args()

    if not os.path.exists(CACHE_PATH):
        print(f"{CACHE_PATH} not found. Run steam_vis.py once to build it.")
        return 1

    df = nl_filter.prepare_dataframe(pd.read_pickle(CACHE_PATH))
    vocab = nl_filter.collect_vocabulary(df)

    cases = [c for c in CASES
             if args.only.lower() in c["name"].lower()
             or args.only.lower() in c["query"].lower()]
    if not cases:
        print(f"No cases match {args.only!r}")
        return 1

    print(f"{len(cases)} cases, {args.runs} run(s) each, "
          f"{len(cases) * args.runs} API calls, model {nl_filter.MODEL}")
    print(f"{len(df):,} games loaded\n")

    report = {}
    for case in cases:
        runs = [run_once(case, df, vocab) for _ in range(args.runs)]
        passes = sum(r["passed"] for r in runs)
        rate = passes / len(runs)

        mark = "PASS" if rate == 1 else "FAIL" if rate == 0 else "FLAKY"
        counts = [r["results"] for r in runs]
        spread = f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}"
        suffix = f"  [{passes}/{len(runs)}]" if args.runs > 1 else ""
        print(f"{mark:5} {case['name']:26} {spread:>10} games{suffix}")

        # Show every distinct failure seen across the runs.
        seen = []
        for r in runs:
            for f in r["failures"]:
                if f not in seen:
                    seen.append(f)
        for f in seen:
            print(f"          {f}")

        report[case["name"]] = {
            "query": case["query"],
            "pass_rate": rate,
            "results": counts,
            "failures": seen,
            "seconds": round(statistics.mean(r["seconds"] for r in runs), 2),
        }

    overall = sum(r["pass_rate"] for r in report.values()) / len(report)
    print(f"\npass rate {overall:.0%}  ({sum(1 for r in report.values() if r['pass_rate'] == 1)}"
          f"/{len(report)} cases fully passing)")

    previous = load_previous()
    if previous:
        prev_cases = previous["cases"]
        regressed = [n for n, r in report.items()
                     if n in prev_cases and r["pass_rate"] < prev_cases[n]["pass_rate"]]
        improved = [n for n, r in report.items()
                    if n in prev_cases and r["pass_rate"] > prev_cases[n]["pass_rate"]]
        print(f"\ncompared with {previous['label']} (pass rate {previous['pass_rate']:.0%})")
        for n in regressed:
            print(f"  REGRESSED {n}  {prev_cases[n]['pass_rate']:.0%} -> {report[n]['pass_rate']:.0%}")
        for n in improved:
            print(f"  IMPROVED  {n}  {prev_cases[n]['pass_rate']:.0%} -> {report[n]['pass_rate']:.0%}")
        if not regressed and not improved:
            print("  no change")

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        label = time.strftime("%Y-%m-%d_%H-%M-%S")
        path = RESULTS_DIR / f"{label}.json"
        path.write_text(json.dumps({
            "label": label,
            "model": nl_filter.MODEL,
            "runs_per_case": args.runs,
            "pass_rate": overall,
            "cases": report,
        }, indent=2), encoding="utf-8")
        print(f"\nsaved to {path}")

    return 0 if overall == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
