from itertools import combinations
from typing import List, Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
from colorama import Fore, Style
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
from tabulate import tabulate


def getPreProcessingDF(path: str, only_speed_adjustment=False) -> pd.DataFrame:
    """
    Filter out long-duration and mid-start videos

    @param path: Path of raw data
    @param only_speed_adjustment: Return data only for speed-adjusting users if True (default: False)
    @return: Pre-processed dataframe
    """
    df = pd.read_excel(path, sheet_name=None, index_col=0, engine="openpyxl")
    all_df = pd.DataFrame()

    for pid, sheet_df in df.items():
        sheet_df = sheet_df.copy()
        sheet_df["PID"] = pid
        all_df = pd.concat([all_df, sheet_df])

    # remove long-duration(longer than upper-fence) videos
    q1, q3 = all_df["Duration"].quantile(0.25), all_df["Duration"].quantile(0.75)
    IQR = q3 - q1
    upper_fence = q3 + 1.5 * IQR  # 2,179 seconds

    if only_speed_adjustment:
        speed_list = [
            "P5",
            "P11",
            "P13",
            "P14",
            "P16",
            "P17",
            "P18",
            "P22",
            "P23",
            "P24",
            "P25",
        ]
        all_df = all_df[all_df["PID"].isin(speed_list)]

    exclude_long_video = all_df[all_df["Duration"] <= upper_fence]

    # remove mid-start videos
    exclude_midstart_video = exclude_long_video[exclude_long_video["Startmid"] == False]

    return exclude_midstart_video


def getStatistics(
    cols: List[str],
    **kwargs: pd.DataFrame,
) -> None:
    """
    Computes statistics for specified columns in each DataFrame passed via kwargs.

    @param cols: List of column names to compute statistics for.
    @param kwargs: DataFrames to compute statistics from.
    @return: None. Prints Abandonment Rate, Dwell Time, Abandonment Point, and Satisfaction in tabulated form.
    """
    data, headers = [], []

    # Header row
    headers = (
        ["---"] + [f"{k}\n(N={len(df)})" for k, df in kwargs.items()] + ["Statistics"]
    )
    data.append(headers)

    for col in cols:
        row = [col]
        stats_dict = {}

        for k, df in kwargs.items():
            col_data = df[col]

            if col == "Likert":
                valid = col_data[col_data >= 1]
                m, mdn, sd = valid.mean(), valid.median(), valid.std()
                row.append(f"M={m:.3f}\nMdn={mdn:.3f}\nSD={sd:.3f}\n(N={len(valid)})")
                stats_dict[k] = valid.to_numpy()

            elif col == "Abandonment":
                abandon = (df["Abandonment"] == True).sum()
                total = len(df)
                stats_dict[k] = [abandon, total - abandon]
                rate = (abandon / total) * 100
                row.append(f"N={abandon}\n{rate:.5f}%")

            else:
                m, mdn, sd = col_data.mean(), col_data.median(), col_data.std()
                row.append(f"M={m:.3f}\nMdn={mdn:.3f}\nSD={sd:.3f}")
                stats_dict[k] = col_data.to_numpy()

        # Statistical test
        if col == "Abandonment":
            d2 = pd.DataFrame(stats_dict, index=["Abandon", "No Abandon"])
            chi2, p, _, _ = stats.chi2_contingency(d2)
            row.append(f"p={p:.4f}\nstat={chi2:.4f}")
        else:
            h, p = stats.kruskal(*stats_dict.values())
            row.append(f"p={p:.4f}\nstat={h:.4f}")

        data.append(row)

    print(tabulate(data, headers="firstrow", tablefmt="grid"))


def kruskalTest(col: str, **kwargs: pd.DataFrame) -> None:
    """
    Performs the Kruskal-Wallis test on the specified column across the given DataFrames.
    If the result is statistically significant, displays post-hoc Dunn test results using tabulate.

    @param col: The column to perform the test on.
    @param kwargs: DataFrames to include in the test, passed as keyword arguments.
    @return: None. If the test is significant, post-hoc results are printed in tabulated format.
    """
    data = {}

    # Extract column data for each group
    for group_name, df in kwargs.items():
        if col == "Likert":
            valid_values = df[df[col] >= 1][col]
        else:
            valid_values = df[col]
        data[group_name] = valid_values.to_numpy().flatten()

    # Perform Kruskal-Wallis test
    stat, p_value = stats.kruskal(*data.values())

    if p_value >= 0.05:
        print(
            f"No statistically significant difference.\nH = {stat:.4f}, p = {p_value:.4f}"
        )
        return
    else:
        print(
            f"Statistically significant difference found.\nH = {stat:.4f}, p = {p_value:.4f}"
        )

    # Prepare data for post-hoc Dunn's test
    combined_data = pd.DataFrame(
        {
            col: np.concatenate(list(data.values())),
            "Group": np.repeat(list(data.keys()), [len(v) for v in data.values()]),
        }
    )

    # Perform Dunn’s post-hoc test with Bonferroni correction
    dunn_result = sp.posthoc_dunn(
        combined_data, val_col=col, group_col="Group", p_adjust="bonferroni"
    )

    # Reorder rows and columns to match input order
    ordered_groups = list(kwargs.keys())
    dunn_result = dunn_result.loc[ordered_groups, ordered_groups]

    # Highlight significant p-values in red
    dunn_result = dunn_result.applymap(
        lambda x: f"{Fore.RED}{x:.4f}{Style.RESET_ALL}" if x <= 0.05 else f"{x:.4f}"
    )  # type: ignore

    # Print table
    print(tabulate(dunn_result, headers="keys", tablefmt="grid"))


def mannWhitneyUTest(col, **kwargs):
    """
    Performs the Mann-Whitmey U test on the specified column across the two DataFrames.

    @param col: specified column
    @param kwargs: two DataFrames
    @return: None, Display result of Mann-Whitmey U test.
    """

    data = {k: df[col].to_numpy() for k, df in kwargs.items()}
    group1, group2 = data.values()

    stat, p_value = stats.mannwhitneyu(x=group1, y=group2)

    if p_value >= 0.05:
        print(
            f"No statistically significant difference. mann_stat(U) = {stat:.4f}, p-value = {p_value:.4f}"
        )
    else:
        print(
            f"Statistically significant difference found. mann_stat(U) = {stat:.4f}, p-value = {p_value:.4f}"
        )


def compareAbandonRate(**kwargs: pd.DataFrame) -> None:
    """
    Performs a statistical test to examine whether there is a significant difference in Abandonment Rates
    across two or more DataFrames. If there are three or more groups and the chi-square test is significant,
    post-hoc results are also displayed.

    @param kwargs: DataFrames to compare Abandonment Rates between.
    @return: None. Prints the result of the chi-square test. If significant and three or more groups are provided,
            post-hoc results are displayed.
    """
    data = {}  # {group_name: [Abandon count, No Abandon count]}
    sample_sizes = {}  # sample size for each group

    # Calculate abandon counts for each group
    for group, df in kwargs.items():
        abandon = len(df[df["Abandonment"] == True])
        no_abandon = len(df) - abandon
        data[group] = [abandon, no_abandon]
        sample_sizes[group] = len(df)

    groups = list(data.keys())
    df_combined = pd.DataFrame(data, index=["Abandon", "No Abandon"])

    # Perform chi-square test
    chi2, p, dof, _ = stats.chi2_contingency(df_combined)
    if p >= 0.05:  # type: ignore
        print(
            f"\nNo statistically significant difference.\nchi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}"
        )
        return

    print(
        f"\nStatistically significant difference found.\nchi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}"
    )

    # If only two groups, no need for post-hoc
    if len(groups) == 2:
        print("\nOnly two groups — post-hoc analysis is not necessary.")
        return

    # Post-hoc analysis for 3 or more groups
    group_combinations = list(combinations(groups, 2))
    p_values = []

    # Perform pairwise chi-square tests
    for group1, group2 in group_combinations:
        sub_df = df_combined[[group1, group2]]
        chi2, p_val, _, _ = stats.chi2_contingency(sub_df)
        p_values.append(p_val)

    # Bonferroni correction
    _, corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method="bonferroni")

    # Store adjusted p-values in symmetric matrix
    p_matrix = pd.DataFrame(index=groups, columns=groups, dtype="object")
    for idx, (group1, group2) in enumerate(group_combinations):
        adj_p = corrected_pvals[idx]
        formatted = (
            f"{Fore.RED}{adj_p:.4f}{Style.RESET_ALL}"
            if adj_p <= 0.05
            else f"{adj_p:.4f}"
        )
        p_matrix.loc[group1, group2] = formatted
        p_matrix.loc[group2, group1] = formatted

    # Format row/column labels to include sample size
    formatted_index = [f"{g}\n(N={sample_sizes[g]})" for g in groups]
    p_matrix.index = formatted_index  # type: ignore
    p_matrix.columns = formatted_index

    # Print result table
    print("\nPost-hoc results (Bonferroni-corrected p-values):")
    print(
        tabulate(
            p_matrix,  # type: ignore
            headers="keys",
            tablefmt="grid",
            colalign=("center",) * len(groups),
        )
    )


def compareSkipRate(
    key: Literal["Kind_Second", "Kind_Scrubbing", "Dist_Forward", "Dist_Backward"],
    pb: pd.DataFrame,
    enter: pd.DataFrame,
    gaming: pd.DataFrame,
    comedy: pd.DataFrame,
    sports: pd.DataFrame,
    music: pd.DataFrame,
) -> None:
    """
    Compare the usage proportions of skip types and skip directions across the top 6 categories.

    @param key: The comparison key must be one of: "Kind_Second", "Kind_Scrubbing", "Dist_Forward", or "Dist_Backward".
    @return None. If the test is significant, post-hoc results are printed in tabulated format.
    """

    df_list = [
        ("People", pb),
        ("Entertainment", enter),
        ("Gaming", gaming),
        ("Comedy", comedy),
        ("Sports", sports),
        ("Music", music),
    ]

    # 카테고리마다 Skip Type, Skip Direction 따로 저장
    category_dfs_kind, category_dfs_dist = {}, {}

    for category, df in df_list:
        category_dfs_kind[f"{category}_second"] = df[
            ((df["Is10SecondsBackward"]) | (df["Is10SecondsForward"]))
            & (~df["IsScrubbingBackward"])
            & (~df["IsScubbingForward"])
        ]
        category_dfs_kind[f"{category}_scrubbing"] = df[
            ((df["IsScrubbingBackward"]) | (df["IsScubbingForward"]))
            & (~df["Is10SecondsBackward"])
            & (~df["Is10SecondsForward"])
        ]
        category_dfs_kind[f"{category}_both"] = df[
            ((df["Is10SecondsBackward"]) | (df["Is10SecondsForward"]))
            & ((df["IsScrubbingBackward"]) | (df["IsScubbingForward"]))
        ]
        category_dfs_kind[f"{category}_no"] = df[
            (~df["Is10SecondsBackward"])
            & (~df["Is10SecondsForward"])
            & (~df["IsScrubbingBackward"])
            & (~df["IsScubbingForward"])
        ]

        category_dfs_dist[f"{category}_forward"] = df[
            ((df["IsScubbingForward"]) | (df["Is10SecondsForward"]))
            & (~df["Is10SecondsBackward"])
            & (~df["IsScrubbingBackward"])
        ]
        category_dfs_dist[f"{category}_backward"] = df[
            ((df["IsScrubbingBackward"]) | (df["Is10SecondsBackward"]))
            & (~df["IsScubbingForward"])
            & (~df["Is10SecondsForward"])
        ]
        category_dfs_dist[f"{category}_both"] = df[
            ((df["Is10SecondsBackward"]) | (df["IsScrubbingBackward"]))
            & ((df["Is10SecondsForward"]) | (df["IsScubbingForward"]))
        ]
        category_dfs_dist[f"{category}_no"] = df[
            (~df["Is10SecondsBackward"])
            & (~df["Is10SecondsForward"])
            & (~df["IsScrubbingBackward"])
            & (~df["IsScubbingForward"])
        ]

    if key.startswith("Kind_"):
        suffix = key.replace("Kind_", "").lower()
        target_dict = category_dfs_kind
        total_keys = ["second", "scrubbing", "both", "no"]

    elif key.startswith("Dist_"):
        suffix = key.replace("Dist_", "").lower()
        target_dict = category_dfs_dist
        total_keys = ["forward", "backward", "both", "no"]

    else:
        print("Invalid key.")
        return

    categories = [name for name, _ in df_list]
    contingency = []
    sample_sizes = {}

    print(f"Proportion of '{suffix}' usage by category:")
    print("-" * 50)
    print(f"{'Category':<15} {'Count':>7} / {'Total':<7} = {'Ratio (%)':>10}")

    for cat in categories:
        count = len(target_dict[f"{cat}_{suffix}"])
        total = sum(
            len(target_dict[f"{cat}_{k}"])
            for k in total_keys
            if f"{cat}_{k}" in target_dict
        )
        not_count = total - count
        ratio = (count / total) * 100 if total > 0 else 0

        print(f"{cat:<15} {count:>7} / {total:<7} = {ratio:>9.2f}%")
        contingency.append([count, not_count])
        sample_sizes[cat] = total

    contingency = np.array(contingency)
    chi2, p, dof, _ = stats.chi2_contingency(contingency)

    if p >= 0.05:  # type: ignore
        print(
            f"\nNo statistically significant difference.\nchi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}"
        )
        return

    print(
        f"\nStatistically significant difference found.\nchi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}"
    )

    if len(categories) == 2:
        print("\nOnly two groups — post-hoc analysis is not necessary.")
        return

    group_combinations = list(combinations(categories, 2))
    p_values = []

    for g1, g2 in group_combinations:
        c1 = len(target_dict[f"{g1}_{suffix}"])
        t1 = sum(
            len(target_dict[f"{g1}_{k}"])
            for k in total_keys
            if f"{g1}_{k}" in target_dict
        )
        c2 = len(target_dict[f"{g2}_{suffix}"])
        t2 = sum(
            len(target_dict[f"{g2}_{k}"])
            for k in total_keys
            if f"{g2}_{k}" in target_dict
        )

        sub_table = np.array([[c1, t1 - c1], [c2, t2 - c2]])
        chi2, p_val, _, _ = stats.chi2_contingency(sub_table)
        p_values.append(p_val)

    _, corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method="bonferroni")

    p_matrix = pd.DataFrame(index=categories, columns=categories, dtype="object")
    for idx, (g1, g2) in enumerate(group_combinations):
        adj_p = corrected_pvals[idx]
        formatted = f"\033[91m{adj_p:.4f}\033[0m" if adj_p <= 0.05 else f"{adj_p:.4f}"
        p_matrix.loc[g1, g2] = formatted
        p_matrix.loc[g2, g1] = formatted

    for g in categories:
        p_matrix.loc[g, g] = ""

    formatted_index = [f"{g}\n(N={sample_sizes[g]})" for g in categories]
    p_matrix.index = formatted_index  # type: ignore
    p_matrix.columns = formatted_index

    print("\nPost-hoc results (Bonferroni-corrected p-values):")
    print(
        tabulate(
            p_matrix,  # type: ignore
            headers="keys",
            tablefmt="grid",
            colalign=("center",) * len(categories),
        )
    )
