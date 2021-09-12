from surprise import Dataset, Reader, NMF
from surprise import accuracy

from data_process.data_reader.data_enum import DataSetEnum
from data_process.data_reader.process_raw_data import load_processed_data, split_train_dev_test_data
from tool.best_config_helper import read_best_config
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, create_path

SURPRISE_BEST_CONFIG_PATH = "out/tune/best_config/surprise.json"


def load_data(data_set: DataSetEnum):
    df = load_processed_data(data_set)
    reader = Reader(rating_scale=(1, 5))
    train, dev, test = split_train_dev_test_data(df)

    train = train[train["rating"] != 0.]
    train = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)
    train_set = train.construct_trainset(train.raw_ratings)

    dev = Dataset.load_from_df(dev[['userID', 'itemID', 'rating']], reader)
    dev_set = dev.construct_testset(dev.raw_ratings)

    test = Dataset.load_from_df(test[['userID', 'itemID', 'rating']], reader)
    test_set = test.construct_testset(test.raw_ratings)

    return train_set, dev_set, test_set


def run_exp(algo, train_data, test_data, data_set=None):
    algo.fit(train_data)
    predictions = algo.test(test_data)
    mse = accuracy.mse(predictions, verbose=False)
    if data_set is not None:
        logger.info("Writing result...")
        result_path = ROOT_DIR.joinpath(f"out/predict/{algo.__class__.__name__}_{data_set}.csv")
        create_path(result_path)
        with open(result_path, "w", encoding="utf-8") as f:
            mse_array = [float((true_r - est) ** 2) for (_, _, true_r, est, _) in predictions]
            for i in mse_array:
                f.write(f"{i}\n")
    return mse


def main():
    for data_set in [
        DataSetEnum.Appliances,
        DataSetEnum.Digital_Music,
        DataSetEnum.Luxury_Beauty,
        DataSetEnum.Prime_Pantry,
        DataSetEnum.Yelp,
    ]:
        logger.info(f"---------------------- Evaluating on {data_set} ----------------------")
        train_data, _, test_data = load_data(data_set)
        config = read_best_config(NMF, data_set)
        algo = NMF(n_factors=50,
                   lr_bu=config["lr"], lr_bi=config["lr"],
                   reg_pu=config["reg_p"], reg_qi=config["reg_p"],
                   reg_bu=config["reg_b"], reg_bi=config["reg_b"])
        mse = run_exp(algo, train_data, test_data, data_set)
        logger.info(f"{algo.__class__.__name__}:{mse}")


if __name__ == '__main__':
    main()
