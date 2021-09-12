import json

from tqdm import tqdm

from tool.path_helper import ROOT_DIR

with open(ROOT_DIR.joinpath("data/yelp/yelp_academic_dataset_review.json"), "r", encoding="utf-8", errors='ignore') as in_f:
    with open(ROOT_DIR.joinpath("data/raw_data/Yelp.json"), "w", encoding="utf-8") as out_f:
        temp_lines = []
        for line in tqdm(in_f):
            json_obj = json.loads(line)

            review = {
                "reviewerID": json_obj["user_id"],
                "asin": json_obj["business_id"],
                "reviewText": json_obj["text"],
                "overall": float(json_obj["stars"]),
            }

            out_f.write(json.dumps(review) + "\n")
