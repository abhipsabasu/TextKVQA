import json


def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k, lower_keys(v)) for k, v in x.items())
    elif isinstance(x, str):
        return x.lower()
    else:
        return x


with open("data/QA-scene.json") as f:
    question_json = json.load(f)

final = lower_keys(question_json)
print(final)
