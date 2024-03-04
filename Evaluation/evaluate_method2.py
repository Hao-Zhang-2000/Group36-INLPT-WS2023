import glob
import pandas as pd
import json

outputs = []
for file in glob.glob("../output/*.json"):

    output = pd.DataFrame(json.load(open(file, "r")))
    if output["eval_score_GPT3.5"][0] == "YES" or output["eval_score_GPT3.5"][0] == "NO":
        output["settings"] = file
        outputs.append(output)

result = pd.concat(outputs)

result = pd.concat(outputs)

result["eval_score_GPT3.5"] = result["eval_score_GPT3.5"].apply(
    lambda x: 1 if x == "YES" else 0
)

average_scores = result.groupby("settings")["eval_score_GPT3.5"].mean()
with open("average_scores2.json", "w") as f:
    json.dump(average_scores.to_dict(), f)

print(average_scores.sort_values())