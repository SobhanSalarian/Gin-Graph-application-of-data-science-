"""
LLM-as-judge check of the genre labels behind the new TMDb datasets. The labels
are TMDb's own, so nobody has reviewed them for this project; here an LLM acts
as an independent annotator, and we measure how well it agrees with TMDb.

The judge sees a film's title and top-billed cast, never its TMDb label, and has
no tools, so it answers from the title and cast alone. Films go in shuffled
batches. It can also answer UNSURE, which counts as disagreement in raw agreement
and is left out of Cohen's kappa.

Calls go through the Claude Code CLI (`claude -p`), which must be logged in.
Answers are saved after each batch, and a rerun only asks about unanswered films.

Usage:
    python evaluation/llm_judge.py --mode binary
    python evaluation/llm_judge.py --mode multi
"""
import argparse
import json
import random
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "llm_judge"
SOURCE_NAMES = {
    "binary": "IMDBBINARY-2016",
    "multi": "IMDBMULTI-2016",
}

PROMPT = """You are annotating films by genre. For each film below, choose the single \
genre that best describes it from: {genres}. Answer UNSURE if you cannot tell from \
the title and cast.

Reply with only a JSON object mapping each film's id to your answer, for example \
{{"0": "{example}", "1": "UNSURE"}}.

Films:
{films}"""


def ask_judge(prompt, model, workdir):
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--tools", "", "--no-session-persistence"],
        input=prompt, capture_output=True, text=True, encoding="utf-8",
        cwd=workdir, shell=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"judge call failed:\n{proc.stderr[-2000:]}")
    found = re.search(r"\{.*\}", proc.stdout, re.DOTALL)
    if not found:
        raise RuntimeError(f"no JSON in judge reply:\n{proc.stdout[-2000:]}")
    return json.loads(found.group())


def cohen_kappa(pairs):
    """pairs = (tmdb_label, judge_label) for films the judge did not mark UNSURE."""
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    tmdb = Counter(a for a, _ in pairs)
    judge = Counter(b for _, b in pairs)
    expected = sum(tmdb[g] * judge[g] for g in tmdb) / (n * n)
    return (observed - expected) / (1 - expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "multi"], required=True)
    ap.add_argument("--per-genre", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--cast-shown", type=int, default=8)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    source = SOURCE_NAMES[args.mode]
    meta_path = ROOT / "new_data" / "processed" / source / f"{source}_metadata.json"
    meta = json.loads(meta_path.read_text())
    genres = meta["label_order"]

    rng = random.Random(args.seed)
    sample = []
    for genre in genres:
        films = [g for g in meta["graphs"] if g["label"] == genre]
        sample.extend(rng.sample(films, args.per_genre))
    rng.shuffle(sample)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{source}_judgements.json"
    answers = {}
    if out_path.exists():
        answers = {r["movie_id"]: r["judge"] for r in json.loads(out_path.read_text())["films"]}

    def save():
        rows = [{"movie_id": f["movie_id"], "title": f["title"], "tmdb": f["label"],
                 "judge": answers[f["movie_id"]]}
                for f in sample if f["movie_id"] in answers]
        out_path.write_text(json.dumps({"model": args.model, "seed": args.seed,
                                        "genres": genres, "films": rows}, indent=2))
        return rows

    # run outside the repo so the judge cannot pick up any project context
    with tempfile.TemporaryDirectory() as workdir:
        pending = [f for f in sample if f["movie_id"] not in answers]
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            films = "\n".join(
                f"{i}: {f['title']} | cast: {', '.join(f['actor_names'][:args.cast_shown])}"
                for i, f in enumerate(batch)
            )
            reply = ask_judge(
                PROMPT.format(genres=", ".join(genres), example=genres[0], films=films),
                args.model, workdir,
            )
            for i, f in enumerate(batch):
                answer = reply.get(str(i), "UNSURE")
                answers[f["movie_id"]] = answer if answer in genres else "UNSURE"
            save()
            print(f"{min(start + args.batch_size, len(pending))}/{len(pending)} judged", flush=True)

    rows = save()
    decided = [(r["tmdb"], r["judge"]) for r in rows if r["judge"] != "UNSURE"]
    agree = sum(a == b for a, b in decided)
    print(f"\n{source}, judge {args.model}, {len(rows)} films")
    print(f"raw agreement {100 * agree / len(rows):.1f}%")
    print(f"unsure {len(rows) - len(decided)}")
    print(f"agreement when decided {100 * agree / len(decided):.1f}%")
    print(f"Cohen's kappa {cohen_kappa(decided):.3f}")
    print(f"disagreements by TMDb label: {dict(Counter(a for a, b in decided if a != b))}")


if __name__ == "__main__":
    main()
