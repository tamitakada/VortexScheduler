import os
import argparse
import json
import pandas as pd


TABLE_CSS = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&icon_names=arrow_upward" />
    <style>
        .material-symbols-outlined {
            font-variation-settings:
            'FILL' 0,
            'wght' 400,
            'GRAD' 0,
            'opsz' 24
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Space Mono';
        }

        html, body {
            width: 100%;
        }

        body {
            padding: 10px;
        }

        table, td, th {
            border: 1px solid black;
        }

        table tr:not(:first-child) th {
            white-space: nowrap;
        }

        tr:nth-child(even), tr:nth-child(even) th:first-child {
            background-color: #f2f2f2;
        }

        tr:nth-child(odd), tr:nth-child(odd) th:first-child {
            background-color: white;
        }

        th {
            font-weight: bold;
            padding: 5px;
            font-size: 11pt;
        }

        td {
            padding: 5px;
            text-align: right;
            font-size: 11pt;
        }

        .table-title-div {
            background-color: #f2cbc9;
            padding: 5px;
        }

        h2 {
            color: #ef2217;
            font-weight: bold;
            font-size: 14pt;
        }

        h3 {
            font-size: 12pt;
        }

        th:first-child {
            position: sticky;
            left: 0;
            border: 1px solid black;
            z-index: 10;
        }

        #to-top-button {
            background-color: rgba(18, 61, 135, 0.7);
            width: 70px; height: 90px;
            position: fixed; right: 20px; bottom: 20px;
            border-radius: 5px;
            display: inline-flex; flex-direction: column; align-items: center; justify-content: center;
            text-decoration: none; color: white; text-align: center; v-align: center;
        }
    
    </style>
"""

def generate_menu(df: pd.DataFrame):
    html_str = f"""
        <div style='display: flex; flex-direction: column; padding-bottom: 20px;'>
            <p style='font-size: 14pt;'><strong>Contents:</strong></p>
    """

    for group in df["group"].unique():
        html_str += f"<a href='#{group.replace(' ', '_').lower()}'>{group}</a>\n"

    return html_str + "</div>"


def generate_comparison_table(df: pd.DataFrame, out_path: str):
    html_str = "<html><head>" + TABLE_CSS + "</head><body>" + generate_menu(df)
    html_str += """
        <a id='to-top-button' href='#'>
            <span class='material-symbols-outlined'>
                arrow_upward
            </span>
            <p>to<br>top</p>
        </a>
    """
    
    for group in df["group"].unique():
        html_str += f"""
        <div id={group.replace(' ', '_').lower()} class='table-title-div'><h2>{group}</h2></div>
        <h3 style='margin: 5px 0;'>Stats across all steps</h3>
        <div style='overflow-x: scroll;'>
        """

        by_job_type_table = """
        <table style='margin-bottom: 10px;'>
            <tr>
                <th>Scheduler</th>
                <th>Throughput (QPS)</th>
                <th>Goodput (QPS)</th>
                <th>Median latency (ms)</th>
                <th>Mean latency (ms)</th>
                <th>Median tardiness (ms)</th>
                <th>Mean tardiness (ms)</th>
                <th>Dropped percent</th>
                <th>Drop rate (QPS)</th>
                <th>Tardy percent</th>
                <th># complete</th>
                <th># dropped</th>
                <th># tardy</th>
            </tr>
        """

        by_task_type_table = """
        <h3 style='margin: 5px 0;'>Stats per step</h3>
        <table style='margin-bottom: 20px;'>
            <tr>
                <th>Scheduler</th>
                <th>Task Type</th>
                <th>Median latency (ms)</th>
                <th>Mean latency (ms)</th>
                <th>Median batch size</th>
                <th>Mean batch size</th>
                <th>Min batch size</th>
                <th>Max batch size</th>
            </tr>
        """

        for i, row in df[df["group"]==group].iterrows():
            scheduler_name = row["scheduler"]

            with open(os.path.join(row["path_to_data"], "stats.json"), "r") as f:
                scheduler_stats = json.loads(f.read())
                f.close()

            batch_df = pd.read_csv(os.path.join(row["path_to_data"], "batch_log.csv"))

            if len(scheduler_stats["clients"]) == 1 and len(scheduler_stats["clients"][0].keys()) == 1:
                scheduler_stats = list(scheduler_stats["clients"][0].values())[0]
                by_job_type_table += f"""<tr>
                    <th style='text-align: left;'>{scheduler_name}</th>
                    <td>{scheduler_stats['throughput_qps']:.2f}</td>
                    <td>{scheduler_stats['goodput_qps']:.2f}</td>
                    <td>{scheduler_stats['median_latency_ms']:.2f}</td>
                    <td>{scheduler_stats['mean_latency_ms']:.2f}</td>
                    <td>{scheduler_stats['median_tardiness_ms']:.2f}</td>
                    <td>{scheduler_stats['mean_tardiness_ms']:.2f}</td>
                    <td>{scheduler_stats['total_num_dropped'] / 100:.2f}</td>
                    <td>{scheduler_stats['drop_rate_qps']:.2f}</td>
                    <td>{scheduler_stats['total_num_tardy'] / 100:.2f}</td>
                    <td>{10000 - scheduler_stats['total_num_dropped']}</td>
                    <td>{scheduler_stats['total_num_dropped']}</td>
                    <td>{scheduler_stats['total_num_tardy']}</td>
                </tr>"""

                task_types = list(map(tuple, batch_df[['workflow_id', 'task_id']].drop_duplicates().values))
                for task_type in task_types:
                    batch_df_tt = batch_df[(batch_df["workflow_id"]==task_type[0]) & (batch_df["task_id"]==task_type[1])]
                    batch_exec_times = batch_df_tt['end_time']-batch_df['start_time']
                    by_task_type_table += f"""<tr>
                        <th style='text-align: left;'>{scheduler_name}</th>
                        <th style='text-align: left;'>{int(task_type[0])}, {int(task_type[1])}</td>
                        <td>{batch_exec_times.median():.2f}</td>
                        <td>{batch_exec_times.mean():.2f}</td>
                        <td>{batch_df_tt['batch_size'].median()}</td>
                        <td>{batch_df_tt['batch_size'].mean():.2f}</td>
                        <td>{batch_df_tt['batch_size'].min()}</td>
                        <td>{batch_df_tt['batch_size'].max()}</td>
                    </tr>"""
        
        by_job_type_table += "</table></div>"
        by_task_type_table += "</table></div>"

        html_str += by_job_type_table + by_task_type_table

    html_str += "</body></html>"

    with open(os.path.join(out_path, "comparison_summary.html"), "w") as f:
        f.write(html_str)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--table-data-path", type=str, required=True)
    parser.add_argument("-o", "--out", type=str, default="results", help="Path to output directory")
    
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.table_data_path)
    generate_comparison_table(df, args.out if args.out else "")