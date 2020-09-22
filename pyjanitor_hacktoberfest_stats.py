# Standard libraries
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Tuple

# Third-party libraries
import requests
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from plotly import subplots, offline

# Local Libraries

# TODOS
# - Unit tests


def _setup_data_dir() -> None:
    """
    Helper function to create necessary data directory and files.
    """
    data_dir = Path("data/")

    if not data_dir.exists():
        print(f"\nCreating data directory: {data_dir}")
        data_dir.mkdir()
        empty_df = pd.DataFrame(
            columns=[
                "issues_open",
                "issues_closed",
                "issues_total",
                "prs_open",
                "prs_closed",
                "prs_total",
                "hacktoberfest_issues_open",
                "hacktoberfest_issues_closed",
                "hacktoberfest_issues_total",
            ]
        )
        empty_df.to_csv(data_dir.joinpath("pyjanitor_hacktoberfest_2020.csv"))


def _get_total_page_count(url: str) -> int:
    """
    Helper function to get total number of pages for scrapping GitHub.

    Notes:
        - This page count is equivalent to the sum of total Issues
          and Pull Requests (regardless of open of closed status).
        - This should be the same number as navigating to the Issues
          page of a repository and deleting the ``is:issue is:open``
          filter from the search bar and pressing Enter. The sum of
          ``Open`` and ``Closed`` should be the same as page count
          returned here.
    """
    # Order matters here for last_url ensue ``page`` follows ``per_page``
    params = {"state": "all", "per_page": 1, "page": 0}
    response = requests.get(url=url, params=params)  # type: ignore[arg-type]

    if not response.ok:
        raise requests.exceptions.ConnectionError(f"Failed to connect to {url}")

    last_url_used = response.links["last"]["url"]
    start_index = last_url_used.find("&page=")
    page_count = int(last_url_used[(start_index + len("&page=")) :])

    return page_count


def _calculate_needed_batches(page_count: int) -> int:
    """
    Helper function to determine batch size for GitHub scrapping.
    This accounts for the GitHub API use of pagination.
    This accounts for scrapping 100 pages per batch.
    """
    n_batches = page_count // 100 + 1

    return n_batches


def scrape_github(url: str, needed_batches: int) -> List[Dict[str, Any]]:
    """
    Scrape GitHunb repo and collect all Issues and Pull Requests.

    Notes:
        - GitHub treats Pull Requests as Issues.
        - ``state: 'all'`` is used to collect both Open and Closed
            Issues/PRs.
        - GitHub API uses pagination, which means it returns only a
          chunk of information in a single request. Thus, need to
          set how many records will be included per page and the
          page number. Here, we collect 100 items per request.
    """
    scraped_data = []

    # Pagination needs to start at 1 since page 0 and page 1 are duplicates
    # Account for this by using range from [1,batches+1] instead of [0,batches]
    for i in tqdm(range(1, needed_batches + 1), desc="\tGitHub Scrape", ncols=75):
        params = {"state": "all", "per_page": 100, "page": i}
        response = requests.get(url=url, params=params)
        scraped_data.extend(response.json())

    return scraped_data


def create_metadata_df(scraped_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary dataframe of all info scraped from GitHub.
    """
    df = pd.DataFrame(scraped_data)

    keep_cols = [
        "number",
        "title",
        "state",
        "labels",
        "created_at",
        "updated_at",
        "closed_at",
        "pull_request",
    ]

    df = df[keep_cols]

    df = df.assign(labels=df["labels"].apply(lambda x: [d.get("name") for d in x]))
    df = df.assign(
        hacktober=df["labels"].apply(lambda x: "hacktoberfest-2020" in x).astype(int)
    )

    return df


def create_topic_dfs(
    metadata_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Helper function to split Issues, PRs, and Hacktoberfest Issues
    into separate, individual dataframes.
    """
    issues_df = metadata_df[metadata_df["pull_request"].isnull()]
    prs_df = metadata_df[~metadata_df["pull_request"].isnull()]
    hack_issues_df = issues_df[issues_df["hacktober"] == 1]

    return issues_df, prs_df, hack_issues_df


def collect_topic_counts(topic_df: pd.DataFrame, topic_str: str) -> Dict[str, int]:
    """
    Counts Open, Closed, and Total items for a provided topic.
    Topics are Issues, PRs, or Hacktoberfest Issues.
    """
    cnts = Counter(topic_df["state"])  # type: ignore[var-annotated]

    # If there is nothing closed, add it as a 0 (hacktoberfest)
    cnts["closed"] = cnts.get("closed", 0)
    cnts["total"] = len(topic_df)

    # Add prefix to counts for easy dataframe creation downstream
    cnts = {f"{topic_str}_{k}": v for k, v in cnts.items()}  # type: ignore[assignment]

    return cnts


def _get_todays_data() -> str:
    """ Helper function to get today's date in YYYY-MM-DD format."""
    return datetime.today().strftime("%Y-%m-%d")


def create_current_cnt_df(
    issue_cnts: Dict[str, int],
    pr_cnts: Dict[str, int],
    hack_issue_cnts: Dict[str, int],
    previous_data_path: Path,
) -> pd.DataFrame:
    """
    Create a dataframe that stores the count data as of ``today``.
    This prepares data for plotting and saving.
    """
    today = _get_todays_data()
    today_cnt_df = pd.DataFrame(
        {**issue_cnts, **pr_cnts, **hack_issue_cnts}, index=[today]
    )

    previous_cnt_df = pd.read_csv(previous_data_path, index_col=0)
    current_cnt_df = pd.concat([previous_cnt_df, today_cnt_df])

    print(f"\tUpdating count data by appending counts from {today}")
    print(f"\tSaving updates to {previous_data_path}")
    current_cnt_df.to_csv(previous_data_path)

    return current_cnt_df


def _make_scatter_trace(
    current_cnt_df: pd.DataFrame, plot_col: str, name: str
) -> go.Scatter:
    """
    Helper function to create the scatter traces for Issues and PRs.
    """
    ORANGE = "#ff580a"
    BLACK = "#080808"

    line_color = ORANGE if "issues" in plot_col else BLACK
    circle_color = BLACK if "issues" in plot_col else ORANGE

    trace = go.Scatter(
        x=pd.to_datetime(current_cnt_df.index, format="%Y-%m-%d"),
        y=current_cnt_df[plot_col],
        mode="lines+markers",
        name=name,
        marker=dict(color=line_color, line=dict(width=2, color=circle_color)),
        showlegend=False,
    )

    return trace


def _make_bar_trace(current_cnt_df: pd.DataFrame, name: str) -> go.Bar:
    """
    Helper function to create the stacked bar traces for Issues and PRs.
    """
    ORANGE = "#ff580a"
    BLACK = "#080808"

    color = ORANGE if name == "Open" else BLACK
    today = _get_todays_data()

    trace = go.Bar(
        name=name,
        x=["Hacktoberfest Issues", "Issues", "Pull Requests"],
        y=current_cnt_df.loc[today][
            [
                f"hacktoberfest_issues_{name.lower()}",
                f"issues_{name.lower()}",
                f"prs_{name.lower()}",
            ]
        ].values,
        marker_color=color,
        width=0.5,
    )

    return trace


def _annotate_scatter(fig: go.Figure, current_cnt_df: pd.DataFrame) -> None:
    """
    Helper function to annotate Issues and PRs on scatter plot.
    Legend is not shown so need annotations to see which line
    pertains to Issues and which line pertains to PRs.
    """
    x_loc = pd.to_datetime(current_cnt_df.iloc[0].name, format="%Y-%m-%d")

    fig.add_annotation(
        x=x_loc,
        y=current_cnt_df.iloc[0].issues_closed,
        text="Issues",
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=x_loc,
        y=current_cnt_df.iloc[0].prs_closed,
        text="Pull Requests",
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=x_loc,
        y=current_cnt_df.iloc[0].issues_open,
        text="Issues",
        row=2,
        col=1,
    )

    fig.add_annotation(
        x=x_loc,
        y=current_cnt_df.iloc[0].prs_open,
        text="Pull Requests",
        row=2,
        col=1,
    )


def _make_table_trace(current_cnt_df: pd.DataFrame) -> go.Table:
    """
    Helper function to convert pandas dataframe to plotly table.
    """
    LIGHT_ORANGE = "#f18d61"
    ORANGE = "#ff580a"
    BLACK = "#080808"

    table_data = current_cnt_df.reset_index().rename(columns={"index": "Date"})

    trace = go.Table(
        header=dict(
            values=[
                "<br>Date",
                "<br>Open Issues",
                "<br>Closed Issues",
                "<br>Total Issues",
                "Open Pull Requests",
                "Closed Pull Requests",
                "Total Pull Requests",
                "Hacktoberfest<br>Open Issues",
                "Hacktoberfest<br>Closed Issues",
                "Hacktoberfest<br>Total Issues",
            ],
            font=dict(color=ORANGE),
            align="center",
            line_color=ORANGE,
            fill_color=BLACK,
        ),
        # Reverse order table data for most recent data at top
        cells=dict(
            values=[table_data[c][::-1].tolist() for c in table_data.columns],
            font=dict(color=BLACK),
            align="center",
            line_color=BLACK,
            fill_color=LIGHT_ORANGE,
        ),
    )

    return trace


def create_hacktoberfest_plot(
    current_cnt_df: pd.DataFrame,
) -> go.Figure:
    """
    Create final plot.
    """
    BLACK = "#080808"

    today = _get_todays_data()

    fig = subplots.make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=(
            "Closed Issues and Pull Requests",
            f"Current Counts as of {today}",
            "Open Issues and Pull Requests",
        ),
        specs=[
            [{}, {"rowspan": 2}],
            [{}, None],
            [{"colspan": 2, "type": "table"}, None],
        ],
    )

    ########################
    # Create Scatter Plots #
    ########################
    scatter_issue_closed = _make_scatter_trace(
        current_cnt_df=current_cnt_df, plot_col="issues_closed", name="Closed Issues"
    )
    fig.add_trace(scatter_issue_closed, row=1, col=1)

    scatter_pr_closed = _make_scatter_trace(
        current_cnt_df=current_cnt_df,
        plot_col="prs_closed",
        name="Closed Pull Requests",
    )
    fig.add_trace(scatter_pr_closed, row=1, col=1)

    scatter_issue_open = _make_scatter_trace(
        current_cnt_df=current_cnt_df, plot_col="issues_open", name="Open Issues"
    )
    fig.add_trace(scatter_issue_open, row=2, col=1)

    scatter_pr_open = _make_scatter_trace(
        current_cnt_df=current_cnt_df, plot_col="prs_open", name="Open Pull Requests"
    )
    fig.add_trace(scatter_pr_open, row=2, col=1)

    fig.update_xaxes(
        tickformat="%m/%d",
        tickmode="array",
        tickvals=current_cnt_df.index[::2],
        tickangle=45,
    )

    max_closed = current_cnt_df.filter(regex="closed").values.max()
    fig.update_yaxes(title_text="Count", range=[0, max_closed + 50], row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    _annotate_scatter(fig=fig, current_cnt_df=current_cnt_df)

    ############################
    # Create Stacked Bar Plots #
    ############################
    # Order matters (add closed first so it appears on bottom)
    closed_trace = _make_bar_trace(current_cnt_df=current_cnt_df, name="Closed")
    fig.add_trace(closed_trace, row=1, col=2)

    open_trace = _make_bar_trace(current_cnt_df=current_cnt_df, name="Open")
    fig.add_trace(open_trace, row=1, col=2)

    ########################
    # Create Summary Table #
    ########################
    table_trace = _make_table_trace(current_cnt_df=current_cnt_df)
    fig.add_trace(table_trace, row=3, col=1)

    fig.update_layout(
        title=dict(
            text="👻 PyJanitor Hacktoberfest 2020 🎃",
            font=dict(color=BLACK, size=24, family="Scary Halloween Font"),
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="middle",
        ),
        barmode="stack",
        height=1000,
        width=1500,
        font_family="Spooky Skeleton",
        font_color=BLACK,
    )

    return fig


def main() -> None:
    START = datetime.now()

    _setup_data_dir()

    URL = "https://api.github.com/repos/ericmjl/pyjanitor/issues"
    PREV_DATA_PATH = Path("data/pyjanitor_hacktoberfest_2020.csv")
    PLOT_DATA_PATH = Path("data/pyjanitor_hacktoberfest_stats.html")

    print("\nScrapping GitHub to find total number of pages...")
    page_cnt = _get_total_page_count(url=URL)
    n_batches = _calculate_needed_batches(page_count=page_cnt)
    print(f"\tFound {page_cnt:,} total pages")
    print(f"\tNeed {n_batches} batches for scrapping")

    print("\nScraping GitHub for issue and pull request data...")
    scraped_data = scrape_github(url=URL, needed_batches=n_batches)

    print("\nCreating global metadata dataframe and topic dataframes...")
    metadata_df = create_metadata_df(scraped_data=scraped_data)
    issues_df, prs_df, hack_issues_df = create_topic_dfs(metadata_df=metadata_df)

    print("\nCollecting topic counts...")
    issue_cnts = collect_topic_counts(topic_df=issues_df, topic_str="issues")
    pr_cnts = collect_topic_counts(topic_df=prs_df, topic_str="prs")
    hack_issue_cnts = collect_topic_counts(
        topic_df=hack_issues_df, topic_str="hacktoberfest_issues"
    )

    if issue_cnts["issues_total"] + pr_cnts["prs_total"] != page_cnt:
        raise ValueError(
            f"Page Count:{page_cnt} != (Issues:{issue_cnts['issues_total']} + PRs:{pr_cnts['prs_total']}) collected"
            + f"... {page_cnt} != {issue_cnts['issues_total'] + pr_cnts['prs_total']}"
        )

    print(
        "\nCreating current count data by aggregating yesterday's counts with today's counts..."
    )
    current_cnt_df = create_current_cnt_df(
        issue_cnts=issue_cnts,
        pr_cnts=pr_cnts,
        hack_issue_cnts=hack_issue_cnts,
        previous_data_path=PREV_DATA_PATH,
    )

    print("\nCreating final Hacktoberfest Plot...")
    hacktoberfest_fig = create_hacktoberfest_plot(current_cnt_df=current_cnt_df)

    offline.plot(hacktoberfest_fig, filename=PLOT_DATA_PATH.as_posix(), auto_open=True)
    print(f"\n Complete: {datetime.now() - START}")


if __name__ == "__main__":
    main()
