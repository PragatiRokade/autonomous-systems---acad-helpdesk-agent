import sys
import pandas as pd


def pct(x):
    return f"{100 * x:.0f}%"


for path in sys.argv[1:]:
    df = pd.read_csv(path)
    print(f"\n===== {path}  ({df['id'].nunique()} questions x {df['run'].nunique()} runs)")
    print("Heuristic pass rate:", pct(df["auto_ok"].mean()))

    for cat, g in df.groupby("category"):
        line = f"  {cat:13} heuristic={pct(g['auto_ok'].mean())}"
        if "accuracy_manual" in df:
            line += f"  accuracy={pct(g['accuracy_manual'].mean())}"
        print(line)

    refuse = df[df["expected_behavior"] == "refuse"]
    answer = df[df["expected_behavior"] == "answer"]
    if len(refuse):
        print("Correct refusals (out-of-scope):", pct(refuse["refused"].mean()))
    if len(answer):
        print("False refusals (answerable but refused):", pct(answer["refused"].mean()))
    if "accuracy_manual" in df:
        print("Overall accuracy (manual):", pct(df["accuracy_manual"].mean()))
    if "grounded_manual" in df:
        ans = df[df["expected_behavior"].isin(["answer", "email_draft"])]
        print("Groundedness (manual, answerable only):", pct(ans["grounded_manual"].mean()))

    print("Final agent used:")
    print(df["final_agent"].value_counts().to_string())

    cols = [c for c in ["refused", "has_citation", "accuracy_manual", "final_agent"] if c in df]
    print("Questions that changed between runs:")
    for qid, g in df.groupby("id"):
        if any(g[c].nunique(dropna=False) > 1 for c in cols):
            print(f"  Q{qid}: {g['question'].iloc[0][:80]}")