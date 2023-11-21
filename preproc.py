import pandas as pd

Path = '/users/celdev/jrabault/POESY/DDPM_gen/data/'
# read all the data
df = pd.read_csv(Path + 'IS_method_labels.csv')
print("df", df)
# get and sort unique Date. Should be a reduction of a factor 128

if "ensemble_id" not in df:
    list_dates_unique = df["Date"].unique().tolist()
    # list_dates_unique.sort()

    # G. get only 1/7 of dates, 516/7 = 73,714285714 ~ 74 : don't know why...
    sample_ensemble = []

    id_ens = 0
    df["ensemble_id"] = 0  # only to check if ensembles is good
    # init the list of ensembles for each sample
    ensembles = [None] * len(df)
    for date in list_dates_unique:  # for each unique date
        for j in range(8):  # for each LeadTime between 0 and 7
            # get idx of all samples with the same date and leadtime
            idx = df.loc[(df['Date'] == date) & (
                df["LeadTime"] == j)].index
            # Compute the list of all these samples
            ensemble_list = df.loc[(df['Date'] == date) & (
                df["LeadTime"] == j), "Name"].tolist()
            # affect this whole `ensemble_list` for every samples id the list of list `ensembles`
            for id in idx:
                ensembles[id] = ensemble_list

            # only to check if ensembles is good
            df.loc[(df['Date'] == date) & (
                df["LeadTime"] == j), "ensemble_id"] = id_ens
            id_ens += 1

    print("df", df[["Name", "ensemble_id"]][0:20])

    print("ensembles", ensembles[0:20])

    if "Unnamed: 0" in df:
        df = df.drop('Unnamed: 0', axis=1)

    df.to_csv(Path + 'IS_method_labels.csv', index=False)

    df = pd.read_csv(Path + 'IS_method_labels.csv')

    print("df", df[["Name", "ensemble_id"]][0:20])

else:

    if "Unnamed: 0" in df:
        df = df.drop('Unnamed: 0', axis=1)

    df.to_csv(Path + 'IS_method_labels.csv', index=False)

    print("df", df)

    group = df.groupby(['ensemble_id']).agg(lambda x: x)

    print("group", len(group["Name"]), len(
        group["Name"][0]), group["Name"][0], group["Name"][2000])
