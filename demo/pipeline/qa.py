from typing import Iterable
import jsonlines


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def save_answers(
    queries: Iterable, results: Iterable, recalls: Iterable,path: str = "data/answers.jsonl"
):
    answers = []
    for query, result ,recall in zip(queries, results,recalls):
        answers.append(
            {"id": query["id"], "query": query["query"], "recall":recall,"answer": result.text}
        )

    # use jsonlines to save the answers
    def write_jsonl(path, content):
        with jsonlines.open(path, "w") as json_file:
            json_file.write_all(content)

    # 保存答案到 data/answers.jsonl
    write_jsonl(path, answers)
