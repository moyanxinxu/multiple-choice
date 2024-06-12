import json

import pandas as pd


def json2jsonl(file_path: str):
    idx = []
    text = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as file:
        track_data = json.load(file)
    for item in track_data:
        for t, l in item["sentence_quality"].items():
            idx.append(item["id"])
            text.append(t)

            temp = [0.0] * 5
            for i in l:
                temp[i] = 1.0
            labels.append(temp)
    data = {"id": idx, "text": text, "labels": labels}
    pd.DataFrame(data).to_json(
        f"{file_path.split('.')[0]}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )


if __name__ == "__main__":
    json2jsonl("track1_train.json")
    json2jsonl("track1_val.json")
