import argparse
import pandas as pd
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument(
        "--full_path",
        type=str,
        default="../DATA/small_256/IS_method_labels.csv",
        help="Full path to the CSV file",
    )
    return parser.parse_args()


def process_data(full_path):
    # read all the data
    df = pd.read_csv(full_path)
    print("df", df)

    if "ensemble_id" not in df:
        list_dates_unique = df["Date"].unique().tolist()
        # list_dates_unique.sort()

        id_ens = 0
        df["ensemble_id"] = 0  # only to check if ensembles is good
        # init the list of ensembles for each sample
        ensembles = [None] * len(df)
        leadtimemax = df["LeadTime"].max() + 1
        for date in list_dates_unique:  # for each unique date
            for j in range(leadtimemax):  # for each LeadTime between 0 and 7
                # get idx of all samples with the same date and leadtime
                idx = df.loc[(df["Date"] == date) & (df["LeadTime"] == j)].index
                # Compute the list of all these samples
                ensemble_list = df.loc[
                    (df["Date"] == date) & (df["LeadTime"] == j), "Name"
                ].tolist()
                # affect this whole `ensemble_list` for every samples id the list of list `ensembles`
                for id in idx:
                    ensembles[id] = ensemble_list

                # only to check if ensembles is good
                df.loc[(df["Date"] == date) & (df["LeadTime"] == j), "ensemble_id"] = (
                    id_ens
                )
                id_ens += 1

        print("df", df[["Name", "ensemble_id"]][0:20])

        print("ensembles", ensembles[0:20])

        if "Unnamed: 0" in df:
            df = df.drop("Unnamed: 0", axis=1)

        # Save to a new file with "_ens" appended to the input file name
        output_file = os.path.splitext(full_path)[0] + "_ens.csv"
        df.to_csv(output_file, index=False)

        df = pd.read_csv(output_file)

        print("df", df[["Name", "ensemble_id"]][0:20])

    else:

        if "Unnamed: 0" in df:
            df = df.drop("Unnamed: 0", axis=1)

        df.to_csv(full_path, index=False)

        print("df", df)

        group = df.groupby(["ensemble_id"]).agg(lambda x: x)

        print(
            "group",
            len(group["Name"]),
            len(group["Name"][0]),
            group["Name"][0],
            group["Name"][2000],
        )


def main():
    args = parse_arguments()
    process_data(args.full_path)


if __name__ == "__main__":
    main()
