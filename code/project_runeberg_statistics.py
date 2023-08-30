import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


def read_pg_stats():
    datasets_folder = "%s/lp_experiments/datasets_large" % os.environ['HOME']
    fp = "%s/project_gut_summary.csv" % datasets_folder
    df = pd.read_csv(fp, index_col=0)
    return df


def read_lp_stats(solved=False):
    datasets_folder = "%s/lp_inference/data" % os.environ['HOME']
    if solved:
        fp = "%s/solved_summary.csv" % datasets_folder
    else:
        fp = "%s/unsolved_summary.csv" % datasets_folder
    df = pd.read_csv(fp, index_col=0)
    return df


def combine_stats(pg_df, lp_df):
    return pd.concat([pg_df, lp_df], axis=0)


# read about p values on wikipedia:
# https://en.wikipedia.org/wiki/P-value
def empirical_p_value(distribution, val):
    result = sum(distribution <= val) / len(distribution)
    return result


def check_empirical_p(pr_df, lp_df, statistic):
    lp_val = lp_df.loc["unsolved", statistic]
    pg_mean = pg_df[statistic].values.mean()
    res = empirical_p_value(pg_df[statistic].values, lp_val)
    return res, lp_val, pg_mean


def make_visuals(pg_df, lp_df, statistic):
    res, lp_val, pg_mean = check_empirical_p(pg_df, lp_df, statistic)
    ax = sns.histplot(pg_df, x=statistic)
    ax.axvline(x=lp_val, ymin=0, ymax=1, color="red")
    ax.axvline(x=pg_mean, ymin=0, ymax=1, color="black")
    ax.vlines(x=[lp_val, pg_mean], ymin=0, ymax=1, colors=["red", "blue"])
    title = "Proportion of statistic %s in PR (Red line LP)" % statistic
    plt.title(title)
    plt.show()


def make_p_values():
    pg_df = read_pg_stats()
    lp_df = read_lp_stats()
    stats = []
    p_values = []
    pg_means = []
    lp_values = []
    for stat in lp_df.columns:
        print(stat)
        res, lp_val, pg_mean = check_empirical_p(pg_df, lp_df, stat)
        stats.append(stat)
        p_values.append(res.round(3))
        lp_values.append(lp_val.round(3))
        pg_means.append(pg_mean.round(3))
    p_df = pd.DataFrame(
        {
            "statistic": stats,
            "project_runeberg_average": pg_means,
            "liber_primus_value": lp_values,
            "one_sided_left_p_value": p_values,
        }
    )
    p_df.to_csv("p_values_three.tsv", sep="\t", index=False)


if __name__ == "__main__":
    pg_df = read_pg_stats()
    lp_df = read_lp_stats()
    make_visuals(pg_df, lp_df, "length_2")
    make_visuals(pg_df, lp_df, "length_2")
