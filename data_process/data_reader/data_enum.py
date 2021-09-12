from enum import Enum


class DataSetEnum(Enum):
    def __str__(self):
        return str(self.value)

    Test_Set = "Test_Set"

    Appliances = "Appliances"
    Digital_Music = "Digital_Music"
    Luxury_Beauty = "Luxury_Beauty"
    Prime_Pantry = "Prime_Pantry"
    Yelp = "Yelp"


class DataTypeEnum(Enum):
    def __str__(self):
        return str(self.value)

    Train = "train"
    Test = "test"
    Dev = "dev"


REVIEW_COUNT = 10
REVIEW_LENGTH = 100
RANDOM_STATE = 42


def get_all_dataset():
    all_dataset = list(DataSetEnum)
    all_dataset.remove(DataSetEnum.Test_Set)
    return all_dataset
