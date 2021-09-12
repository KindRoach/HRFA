from scipy.stats import stats

from tool.path_helper import ROOT_DIR


def read_mse(model_save_name: str):
    mse = []
    result_path = ROOT_DIR.joinpath(f"out/predict/{model_save_name}.csv")
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            mse.append(float(line))
    return mse


def t_test(model_1: str, model_2: str):
    mse_1 = read_mse(model_1)
    mse_2 = read_mse(model_2)
    return stats.ttest_rel(mse_1, mse_2)


def main():
    model_pairs = []
    for pair in model_pairs:
        t, p = t_test(pair[0], pair[1])
        print(f"{pair[0]} - {pair[1]} : {t} - {p}")


if __name__ == '__main__':
    main()
