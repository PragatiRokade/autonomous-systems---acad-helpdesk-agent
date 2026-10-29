import csv
from helpdesk.rag import retrieve

print("distance | category      | expected   | question")
for q in csv.DictReader(open("eval/questions.csv", encoding="utf-8")):
    r = retrieve(q["question"])
    print(f'{r["best_distance"]:.3f}    | {q["category"]:13} | {q["expected_behavior"]:10} | {q["question"][:60]}')