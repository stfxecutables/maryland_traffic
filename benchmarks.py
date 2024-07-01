import openml
import pandas as pd
from openml.tasks import TaskType
from pandas import Series
from tqdm import tqdm

DIDS = [
    43524,
    42132,
    41443,
    42345,
]


if __name__ == "__main__":
    tasks = openml.tasks.list_tasks(
        task_type=TaskType.SUPERVISED_CLASSIFICATION, output_format="dataframe"
    )
    tasks = tasks[tasks["name"].str.lower().str.contains("traffic")]
    dfs = []
    for i in tqdm(range(len(tasks))):
        tid = tasks.iloc[i]["tid"]
        ttid = tasks.iloc[i]["ttid"]
        runs = openml.runs.list_runs(
            task=[TaskType.SUPERVISED_CLASSIFICATION.value],
            output_format="dataframe",
        )
        # below is EXTREMELY slow
        runs = openml.runs.get_runs(runs["run_id"])
        runs = [run for run in runs if run.dataset_id in DIDS]
        accs = Series(
            name="acc", data=[run.evaluations["predictive_accuracy"] for run in runs]
        )
        aurocs = Series(
            name="auc", data=[run.evaluations["area_under_roc_curve"] for run in runs]
        )
        df = pd.concat([accs, aurocs], axis=1)
        dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(
        df.sort_values(by=["acc", "auc"], ascending=False)
        .iloc[:100, :]
        .to_markdown(tablefmt="simple", floatfmt="0.3f")
    )
