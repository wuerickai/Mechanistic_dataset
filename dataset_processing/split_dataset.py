import argparse

import pandas as pd


def save_to_textfile(dataframe, filename):
    with open(filename, "w", encoding="utf-8") as destination:
        for text in dataframe["text"]:
            destination.write(text + "\n")


def split_dataset(input_file, train_file, val_file, test_file):
    """Shuffle reaction lines with seed 1205 and split 89%/1%/10%."""
    with open(input_file, encoding="utf-8") as source:
        lines = [line.rstrip("\r\n") for line in source if line.strip()]
    data = pd.DataFrame(lines, columns=["text"])
    shuffled_data = data.sample(frac=1, random_state=1205).reset_index(drop=True)
    train_end = int(0.89 * len(shuffled_data))
    valid_end = int(0.9 * len(shuffled_data))
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:valid_end]
    test_data = shuffled_data[valid_end:]
    save_to_textfile(train_data, train_file)
    save_to_textfile(val_data, val_file)
    save_to_textfile(test_data, test_file)
    print(f"Train: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle and split reaction TXT into train/validation/test files.")
    parser.add_argument("input_file", help="Input reaction TXT path")
    parser.add_argument("train_file", help="Training output TXT path")
    parser.add_argument("val_file", help="Validation output TXT path")
    parser.add_argument("test_file", help="Test output TXT path")
    args = parser.parse_args()
    split_dataset(args.input_file, args.train_file, args.val_file, args.test_file)
