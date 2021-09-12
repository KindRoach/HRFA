from data_process.data_reader.data_enum import get_all_dataset
from data_process.data_reader.process_raw_data import load_processed_data, get_user_count, get_item_count

print(f"Dataset\t\treviews\tusers\titems\tr/u\tr/i")

for data_set in get_all_dataset():
    # Count reviews per user
    all_data = load_processed_data(data_set)
    review_count = len(all_data)
    user_count = get_user_count(all_data)
    item_count = get_item_count(all_data)

    print(f"{data_set}\t"
          f"{review_count / 1000:.0f}\t"
          f"{user_count / 1000:.0f}\t"
          f"{item_count / 1000:.0f}\t"
          f"{review_count / user_count:.2f}\t"
          f"{review_count / item_count:.2f}\t"
          )
