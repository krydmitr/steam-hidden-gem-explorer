# Steam Hidden Gem Explorer

An interactive visualisation of ~2M Steam games, scoring each one on a "hidden gem"
metric: review score relative to how few people own it. The view can be filtered by
typing a description in plain English, for example "cheap co-op games with good
reviews that aren't too long".

![The explorer filtered by the query "coop horror for 3 people". The model reports
that it filtered on the Horror tags and the Co-op category, and that the player
count could not be used because the dataset does not record it.](docs/screenshot.png)

Built with Dash, Plotly and pandas. Natural-language filtering uses the Anthropic
API.

---

## How the AI integration works

A typed description is sent to the model in a single API call. The model returns a
set of filter parameters. Those parameters are applied to the dataframe with
ordinary pandas code, and the charts redraw.

```
"cheap co-op games with good reviews that aren't too long"
        │
        ▼   one API call
{ price_max: 10, category_groups: [["Co-op"]],
  review_score_min: 0.8, playtime_max_hours: 15 }
        │
        ▼   pandas, no API involved
filtered dataframe ──▶ charts redraw
```

The model translates the description into parameters. It does not receive the game
data and does not select which games are shown. The filtering is done by
`apply_filters()`.

### Structured outputs

The request carries a JSON schema, `nl_filter.FILTER_SCHEMA`, describing 18 filter
parameters. The model's output is constrained to that schema as it is generated, so
the reply is always valid JSON matching the expected shape. There is no instruction
to "reply with JSON" and no cleanup of the returned text.

### Grouped tags and categories

Tags and categories are supplied as groups rather than flat lists. Names within a
group are alternatives, and a game must match every group.

```
[["Medieval"], ["Co-op", "Multiplayer"]]
   medieval  AND  (co-op OR multiplayer)
```

An earlier version used a single flat list of tags. Because that list was treated as
an OR, "medieval games to play with friends" returned 4,104 games, of which 210 were
medieval. The flat list could not express "medieval AND multiplayer". Grouping
resolved this without changes to the prompt.

### Dataset vocabulary in the prompt

The model is not told which tag strings exist in this particular dataset unless the
prompt says so. Without the list it returns plausible but unused tag names, which
match no rows. The system prompt is therefore built at startup from the data itself
and includes the 200 most common tags and every category.

### Value checking

The schema constrains the structure of the reply but not the choice of values. Values
are checked and clamped in `normalise_spec()`. The app also displays the model's
interpretation of the request, including the thresholds it selected, above the
charts.

### Error handling

A missing key, an invalid key, a rate limit, a network failure, an HTTP error, a
model refusal and an unparseable reply are all handled. Each produces a message
below the search box, and the charts continue to show their previous contents.

---

## Setup

```bash
pip install -r requirements.txt
```

The natural-language filter requires an Anthropic API key. Copy the template and
add the key:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-...
```

Keys are created in the [Anthropic Console](https://console.anthropic.com/settings/keys).
This is the developer console and is separate from a Claude.ai subscription, which
does not include API access. The API is prepaid, so credit must be added before a
key will work. A search costs a fraction of a cent.

`.env` is gitignored and should not be committed. `.env.example` is the committed
template.

---

## Running

```bash
python steam_vis.py
```

Then open <http://127.0.0.1:8050/>.

The first run parses the 675 MB `games.json` and caches the cleaned dataframe to
`games_clean.pkl`, which takes a few minutes. Later runs load the cache and start in
a few seconds. The cache is rebuilt automatically if `games.json` changes.

Without a key the app still runs and the tag box still works. The natural-language
box reports that no key is configured.

### Example queries

```
cheap co-op games with good reviews that aren't too long
obscure horror games from before 2015
free multiplayer games that run on Linux
the 20 best-reviewed indie games nobody owns
```

The tag box and the natural-language filter are applied together, so a description
can be narrowed further by tag. Charts are zoomable and each point has a tooltip.

---

## Code layout

| File | Contains |
|---|---|
| `steam_vis.py` | Data cleaning, Dash layout, and the two callbacks |
| `nl_filter.py` | The filter schema, the API call, and the pandas filtering |
| `test_nl_filter.py` | Tests for the filtering logic |
| `eval_nl_filter.py` | Evaluation suite for the model |
| `eval_results/` | Recorded eval runs |

## Evaluation

The tests below check the filtering code. The eval suite checks the model.

```bash
python eval_nl_filter.py            one run of every case
python eval_nl_filter.py --runs 3   three runs, to show variance
python eval_nl_filter.py --only medieval
```

18 queries are sent to the API, the returned specs are applied to the real
dataset, and properties of both the spec and the resulting games are asserted.
Exact specs cannot be compared, since several different specs are correct for
one query, so the assertions are properties. A query naming a price must set a
price bound. A query naming one concept must not add a second. A query for
medieval games must return games that are actually tagged Medieval. Names the
model returns must exist in the data.

Each run is written to `eval_results/` and compared against the previous run,
so a prompt change is reported as a named regression or improvement rather than
a changed total. A run costs roughly one API call per case.

Recorded runs so far:

| Run | Pass rate | Change |
|---|---|---|
| Baseline | 89% (16/18) | Two failures found |
| Fixed vocabulary validation and one bad assertion | 94% (17/18) | `single_player` fixed |
| Added "use these exact strings" to the category list | 100% (18/18, 3 runs each) | `readme_headline` fixed |

The two failures in the baseline were of different kinds, which is the reason
for keeping the record. One was a defect: the model returned the category
"Local Co-op", which does not exist in the data, and inside a group of
alternatives a name that matches nothing has no visible effect. `normalise_spec`
now removes unknown names and records them under `_dropped_names`, and adding
"use these exact strings" to the category list in the prompt stopped the model
producing them. The other was a bad assertion: the case required the
Single-player *category*, and the model answered with the Singleplayer *tag*.
99.6% of the games it returned were single-player, so the answer was correct and
the assertion was testing the mechanism rather than the outcome.

## Tests

```bash
python test_nl_filter.py
```

The tests run without an API key, a network connection or the dataset, since the
filtering is ordinary pandas applied to a small set of rows defined in the file.
They cover the tag, category, price, review, owner, year, playtime, platform and
sort filters, the grouped AND/OR behaviour, and two
input cases: a playtime of `0`, which means "no data" rather than a short game, and
a review score returned as a percentage rather than a fraction.

---

## Dataset

FronkonGames, *Steam Games Dataset*, Kaggle:
<https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data>

Not included in this repository. Download `games.json` into the project folder.
