# Build one config per LEMSS paper dataset (Table 1, RQ1-RQ3) from the Llama-E5-Listwise base config.
# Usage: python configs/make_paper_configs.py [rounds]   (default 30, as in the paper)
#        python configs/make_paper_configs.py test       (2 queries x 3 rounds, into configs/paper_test)
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
ROUNDS = 3 if TEST else int(sys.argv[1]) if len(sys.argv) > 1 else 30

MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma": "google/gemma-2-9b-it",
}
RANKERS = {
    "e5": {"e5": {"model_name": "intfloat/e5-large-unsupervised"}},
    "contriever": {"contriever": {"model_name": "facebook/contriever"}},
    "okapi": {"okapi": {"index_name": "okapi_index"}},
}
# Contextualized Part: listwise = ranked lists of the last 2 rounds, pairwise = pairs from the last 3 rounds
FEEDBACK = {"listwise": (False, 2), "pairwise": (True, 3)}

# The exact "no-copy" Instructional Part is not published; this replaces the sentence that allows
# copying parts of other documents with an explicit ban. Confirm the wording with Haya.
ALLOW_COPY = ("However, you are permitted to copy paste parts of documents that other participants have written, "
              "As long as the originality of your document is remained.")
NO_COPY = ("Do not copy any content, sentences or phrases from documents that other participants have written; "
           "write the edits in your own words.")

# (name, model, ranker, feedback, players, no_copy, research questions)
DATASETS = [
    ("llama_e5_listwise_4players", "llama", "e5", "listwise", 4, False, "RQ1"),
    ("llama_e5_pairwise_4players", "llama", "e5", "pairwise", 4, False, "RQ1"),
    ("llama_e5_listwise", "llama", "e5", "listwise", 5, False, "RQ2,RQ3"),
    ("llama_e5_pairwise", "llama", "e5", "pairwise", 5, False, "RQ2,RQ3"),
    ("llama_e5_listwise_nocopy", "llama", "e5", "listwise", 5, True, "RQ2"),
    ("gemma_e5_listwise", "gemma", "e5", "listwise", 5, False, "RQ2"),
    ("llama_contriever_listwise", "llama", "contriever", "listwise", 5, False, "RQ3"),
    ("llama_contriever_pairwise", "llama", "contriever", "pairwise", 5, False, "RQ3"),
    ("llama_okapi_listwise", "llama", "okapi", "listwise", 5, False, "RQ3"),
    ("llama_okapi_pairwise", "llama", "okapi", "pairwise", 5, False, "RQ3"),
]

base = json.loads((HERE / "llama_e5_listwise.json").read_text(encoding="utf-8"))
assert ALLOW_COPY in next(iter(base["agents"].values()))["prompt_format"]

out_dir = HERE / ("paper_test" if TEST else "paper")
out_dir.mkdir(exist_ok=True)
for name, model, ranker, feedback, players, no_copy, rqs in DATASETS:
    cfg = copy.deepcopy(base)
    if TEST:
        cfg["competition"]["init_docs_path"]["docs_file_path"] = "configs/test_initial_documents.trectext"
    cfg["competition"]["rankers"] = RANKERS[ranker]
    cfg["game"]["rounds"] = ROUNDS
    if players == 4:
        # RQ1 drops the Data Science professor persona to match the 4-student human competition
        del cfg["agents"]["ds-professor"]
    pairwise, depth = FEEDBACK[feedback]
    for agent in cfg["agents"].values():
        agent["llm"]["model_name"] = MODELS[model]
        agent["pairwise"] = pairwise
        agent["depth"] = depth
        if no_copy:
            agent["prompt_format"] = agent["prompt_format"].replace(ALLOW_COPY, NO_COPY)
    (out_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{rqs:8} {name}.json")
