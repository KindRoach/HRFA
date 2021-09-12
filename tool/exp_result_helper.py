import pandas
from tinydb import TinyDB

from tool.path_helper import ROOT_DIR

EVAL_RESULT_PATH = "out/eval/summary"


def report_eval(model_save_name: str, data_set: str, eval_result: dict):
    with TinyDB(ROOT_DIR.joinpath(EVAL_RESULT_PATH + ".json")) as db:
        doc = {"model": model_save_name.split("_")[0], "save_name": model_save_name, "data_set": data_set}
        doc.update(eval_result)
        db.insert(doc)


def export_csv():
    results = []
    with TinyDB(ROOT_DIR.joinpath(EVAL_RESULT_PATH + ".json")) as db:
        for doc in db:
            results.append(doc)

    results = pandas.DataFrame(results)
    results.to_csv(ROOT_DIR.joinpath(EVAL_RESULT_PATH + ".csv"), index=False)


def test():
    for i in range(10):
        report_eval(f"model_{i}", "test", {"loss": 0.1 * i, "bleu": 0.2 * i})


if __name__ == '__main__':
    # test()
    export_csv()
