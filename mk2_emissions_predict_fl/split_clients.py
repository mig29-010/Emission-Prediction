import pandas as pd
import os


os.makedirs("clients", exist_ok=True)


DATASETS = [

    ("1kwh_mk1.csv", 4),

    ("1.5kwh_mk1.csv", 3),

    ("2kwh_mk1.csv", 2)

]


client_counter = 1


for dataset_path, num_splits in DATASETS:

    print(f"\nProcessing {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    total_rows = len(df)

    split_size = total_rows // num_splits

    for i in range(num_splits):

        start_idx = i * split_size

        # Last split gets remaining rows
        if i == num_splits - 1:

            end_idx = total_rows

        else:

            end_idx = (i + 1) * split_size

        client_df = df.iloc[start_idx:end_idx]

        output_path = f"clients/client_{client_counter}.csv"

        client_df.to_csv(output_path, index=False)

        print(
            f"Saved {output_path} "
            f"Rows: {start_idx} → {end_idx}"
        )

        client_counter += 1


print("\nAll client datasets created.")
