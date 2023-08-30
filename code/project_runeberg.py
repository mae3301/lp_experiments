import os
import numpy as np
from tqdm import tqdm
import pandas as pd


UNSOLVED_WORD_COUNT = 2902
SOLVED_WORD_COUNT = 728


# percentile function from stack overflow
def percentile(n):
    def percentile_(x):
        return x.quantile(n)

    percentile_.__name__ = "percentile_{:02.0f}".format(n * 100)
    return percentile_


def clean(filepath, instance, word_count):
    with open(filepath, "r") as fh:
        text = fh.read()
        words = text.split(" ")
        cleaned = [w for w in words if len(w) > 0]
        lengths = [len(w) for w in cleaned]
        if len(lengths) < word_count:
            return None
        # change the start index perhaps
        lengths = lengths[0:word_count]
        return lengths


def process_word_info(lengths, instance):
    my_df = pd.DataFrame({"word_length": lengths})
    res = my_df.value_counts()
    res = res.reset_index()
    res["word_length"] = res["word_length"].astype(str)
    num_elements = len(res)
    add = ["length"] * num_elements
    res["str_append"] = add
    res.word_length = res.str_append.str.cat(res.word_length.values, sep="_")
    res = res.drop("str_append", axis=1)
    res.rename(columns={"word_length": "statistic"}, inplace=True)
    res["val"] = res["count"] / res["count"].sum()
    res = res.drop("count", axis=1)
    agg_df = my_df.agg(
        [
            np.mean,
            np.std,
            np.median,
            np.var,
            np.min,
            np.max,
            percentile(0.75),
            percentile(0.25),
        ]
    )
    agg_df = agg_df.reset_index(names=["statistic"])
    agg_df.rename(columns={"word_length": "val"}, inplace=True)
    total_df = pd.concat([agg_df, res], axis=0)
    total_df["instance"] = instance
    pivoted_df = total_df.pivot(
        index="instance",
        columns="statistic",
        values="val"
    )
    return pivoted_df


def process_word_counts(filepath, instance, word_count):
    lengths = clean(filepath, instance, word_count)
    if lengths is None:
        return None
    res_df = process_word_info(lengths, instance)
    return res_df


def create_results(word_count):
    folder = "%s/project-runeberg/files" % os.environ["HOME"]
    files = os.listdir(folder)
    # small = 100
    i = 0
    tmp_df = pd.DataFrame()
    for filename in tqdm(files):
        # if i > small:
        #    continue
        filepath = os.path.join(folder, filename)
        instance_name = filepath.split('.')[0]
        df = process_word_counts(filepath, instance_name)
        if df is None:
            continue
        i = i + 1
        tmp_df = pd.concat([tmp_df, df], axis=0).fillna(0)
    return tmp_df


if __name__ == "__main__":
    word_count = UNSOLVED_WORD_COUNT
    all_results = create_results(word_count)
    datasets_folder = "%s/lp_experiments/datasets_large" % os.environ['HOME']
    output_fp = "%s/project_gut_summary.csv" % datasets_folder
    all_results.to_csv(output_fp)
