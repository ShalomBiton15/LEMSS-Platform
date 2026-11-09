"""
Compare our LEMSS runs with the datasets published with the LEMSS paper (SIGIR'25), round by round.

For every finished run in outputs/, the script finds the matching published competition
(same LLM, prompt type and ranker), downloads it from GitHub, and computes the paper's
measures on both, using TF-IDF cosine similarity as in the paper (Section 5.2):
  - winner_consecutive_sim  similarity of the round winner to the next round's winner (Fig. 3a)
  - top2_sim                similarity between the two highest ranked documents (Fig. 3b)
  - min_list_sim            minimum similarity between documents of the same ranked list (Fig. 3c)
  - unique_docs             number of distinct documents in the ranked list (Fig. 3d)
  - player_consecutive_sim  similarity of each player's document to its previous version (Fig. 3e)
  - sim_to_initial          similarity of each document to the initial document of the game

Usage (from ~/projects/LEMSS-Platform, with the lemss environment active):
    python configs/compare_lemss.py                 # all finished runs, rounds 1-10
    python configs/compare_lemss.py --rounds 5      # only rounds 1-5
Results go to analysis_results/: one CSV per run, a summary CSV, and plots if matplotlib is installed.
"""
import argparse
import json
import re
import urllib.parse
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASETS_URL = "https://raw.githubusercontent.com/LEMSS2025/LEMSS-Datasets/main/{folder}/{file}"
MODEL_NAMES = {"meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama", "google/gemma-2-9b-it": "Gemma"}
RANKER_NAMES = {"e5": "E5", "contriever": "Contriever", "okapi": "Okapi"}
MEASURES = ["winner_consecutive_sim", "top2_sim", "min_list_sim", "unique_docs",
            "player_consecutive_sim", "sim_to_initial"]


def published_folder(config: dict) -> str:
    """Name of the LEMSS-Datasets folder that was produced with the same setup as this config."""
    agents = list(config["agents"].values())
    model = MODEL_NAMES[agents[0]["llm"]["model_name"]]
    ranker = RANKER_NAMES[next(iter(config["competition"]["rankers"]))]
    prompt = "Pairwise" if agents[0]["pairwise"] else "Listwise"
    players = "" if len(agents) == 5 else f" {len(agents)} Players"
    return f"{model} {ranker} {prompt}{players}"


def load_published(folder: str, cache_dir: Path) -> pd.DataFrame:
    """Download (once) and load a published competition as rows of query_id, round, player, rank, document."""
    local = cache_dir / folder
    local.mkdir(parents=True, exist_ok=True)
    for file in ["documents.trectext", "documents.position"]:
        if not (local / file).exists():
            url = DATASETS_URL.format(folder=urllib.parse.quote(folder), file=file)
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, local / file)

    text = (local / "documents.trectext").read_text(encoding="utf-8")
    docs = {m.group(1): m.group(2).strip()
            for m in re.finditer(r"<DOCNO>(.*?)</DOCNO>\s*<TEXT>(.*?)</TEXT>", text, re.DOTALL)}
    # documents.position holds the rank of every docno; 1 is the highest rank, as in LEMSS
    ranks = dict(line.split() for line in (local / "documents.position").read_text().splitlines() if line.strip())

    rows = []
    for docno, document in docs.items():
        _, round_, query_id, player = docno.split("-")
        rank = ranks.get(docno)
        rows.append({"query_id": int(query_id), "round": int(round_), "player": player,
                     "rank": int(rank) if rank else None, "document": document})
    return pd.DataFrame(rows)


def load_ours(history_file: Path) -> pd.DataFrame:
    """Load our competition_history.csv as rows of query_id, round, player, rank, document."""
    df = pd.read_csv(history_file)
    df = df[["query_id", "round", "player", "rank", "document"]].copy()
    df["query_id"] = df["query_id"].astype(int)
    df["round"] = df["round"].astype(int)
    df["document"] = df["document"].fillna("").astype(str).str.strip()
    return df


def compute_measures(df: pd.DataFrame, vectorizer: TfidfVectorizer, rounds: int) -> pd.DataFrame:
    """Compute every measure per query and round, then average over the queries."""
    rows = []
    for query_id, game in df.groupby("query_id"):
        initial = game[game["round"] == 0]["document"].iloc[0]
        lists = {r: g.sort_values("rank") for r, g in game[game["round"].between(1, rounds)].groupby("round")}
        vectors = {r: vectorizer.transform(g["document"]) for r, g in lists.items()}
        initial_vector = vectorizer.transform([initial])

        for r, ranked in lists.items():
            sims = cosine_similarity(vectors[r])
            row = {"query_id": query_id, "round": r,
                   "top2_sim": sims[0, 1],
                   "min_list_sim": min(sims[i, j] for i, j in combinations(range(len(ranked)), 2)),
                   "unique_docs": ranked["document"].nunique(),
                   "sim_to_initial": cosine_similarity(vectors[r], initial_vector).mean()}

            if r - 1 in lists:
                previous = lists[r - 1]
                # winner of the previous round vs winner of this round
                row["winner_consecutive_sim"] = cosine_similarity(vectors[r - 1][0], vectors[r][0])[0, 0]
                # each player's document vs the same player's document in the previous round
                prev_index = {p: i for i, p in enumerate(previous["player"])}
                row["player_consecutive_sim"] = np.mean([
                    cosine_similarity(vectors[r][i], vectors[r - 1][prev_index[p]])[0, 0]
                    for i, p in enumerate(ranked["player"]) if p in prev_index])
            rows.append(row)

    return pd.DataFrame(rows).groupby("round")[MEASURES].mean()


def find_runs(outputs_dir: Path, rounds: int) -> list:
    """Finished runs in outputs/ that have at least `rounds` rounds, newest run per published dataset."""
    runs = {}
    for config_file in sorted(outputs_dir.glob("*/config.json"), key=lambda f: f.stat().st_mtime):
        history_file = config_file.parent / "competition_history.csv"
        if not history_file.exists():
            continue
        config = json.loads(config_file.read_text())
        try:
            folder = published_folder(config)
        except KeyError:
            continue
        history = pd.read_csv(history_file, usecols=["query_id", "round"])
        if history["round"].max() < rounds or history["query_id"].nunique() < 30:
            print(f"skip {config_file.parent.name}: {history['query_id'].nunique()} queries, "
                  f"{history['round'].max()} rounds")
            continue
        runs[folder] = history_file  # a newer run of the same setup replaces an older one
    return sorted(runs.items())


def plot(results: pd.DataFrame, name: str, out_file: Path):
    """One panel per measure, ours vs published over rounds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib is not installed, skipping plots (pip install matplotlib)")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, measure in zip(axes.flat, MEASURES):
        for source, style in [("ours", "-o"), ("published", "--s")]:
            series = results[results["source"] == source].set_index("round")[measure]
            ax.plot(series.index, series.values, style, label=source)
        ax.set_title(measure)
        ax.set_xlabel("round")
        ax.legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare our LEMSS runs to the published LEMSS datasets")
    parser.add_argument("--outputs", default="outputs", help="folder with LEMSS run outputs")
    parser.add_argument("--rounds", type=int, default=10, help="compare rounds 1..ROUNDS")
    parser.add_argument("--results", default="analysis_results", help="where to write the results")
    parser.add_argument("--cache", default="analysis_results/published_data", help="where to keep downloaded data")
    args = parser.parse_args()

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs = find_runs(Path(args.outputs), args.rounds)
    if not runs:
        print(f"No finished run with 30 queries and {args.rounds} rounds in {args.outputs}/")
        return

    summary = []
    for folder, history_file in runs:
        print(f"\n=== {folder}  ({history_file.parent.name})")
        ours = load_ours(history_file)
        published = load_published(folder, Path(args.cache))
        ours = ours[ours["round"] <= args.rounds]
        published = published[published["round"] <= args.rounds]

        # One shared TF-IDF space for both competitions, so their similarities are comparable
        vectorizer = TfidfVectorizer().fit(pd.concat([ours["document"], published["document"]]))
        results = pd.concat([compute_measures(ours, vectorizer, args.rounds).assign(source="ours"),
                             compute_measures(published, vectorizer, args.rounds).assign(source="published")])
        results = results.reset_index()

        name = folder.lower().replace(" ", "_")
        results.to_csv(results_dir / f"{name}.csv", index=False)
        plot(results, folder, results_dir / f"{name}.png")

        means = results.groupby("source")[MEASURES].mean()
        print(means.round(3).to_string())
        for source, row in means.iterrows():
            summary.append({"dataset": folder, "source": source, **row.to_dict()})

    pd.DataFrame(summary).to_csv(results_dir / "summary.csv", index=False)
    print(f"\nResults written to {results_dir}/")


if __name__ == "__main__":
    main()
