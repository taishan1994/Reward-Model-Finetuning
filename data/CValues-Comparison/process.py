import json

def process(path, mode=""):
    with open(path, "r", encoding="utf-8") as fp:
         data = fp.readlines()
    res = []
    for d in data:
        d = json.loads(d)
        prompt = d["prompt"]
        chosen = d["pos_resp"]
        rejected = d["neg_resp"]
        tmp = {"messages": []}
        tmp["messages"].append({"role":"user", "value": prompt})
        tmp["messages"].append({"role":"assistant", "chosen_value": chosen, "rejected_value": rejected})
        res.append(tmp)

    with open("origin_data/{}.jsonl".format(mode), "w", encoding="utf-8") as fp:
        fp.write("\n".join([json.dumps(i, ensure_ascii=False) for i in res]))


if __name__ == '__main__':
    process("origin_data/train.jsonl", "train")
    process("origin_data/test.jsonl", "test")