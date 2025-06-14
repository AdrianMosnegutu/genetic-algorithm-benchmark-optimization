from typing import Dict, Any
import pandas as pd
from scipy import stats as scipy_stats
from datetime import datetime
import os
import numpy as np


class ReportGenerator:
    """Handles generation of HTML reports for genetic algorithm results"""

    def __init__(self):
        """Initialize the report generator"""
        self.html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Genetic Algorithm Statistical Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .significance {{
            font-weight: bold;
        }}
        .sig-3 {{
            color: #e74c3c;
        }}
        .sig-2 {{
            color: #e67e22;
        }}
        .sig-1 {{
            color: #f1c40f;
        }}
        .sig-ns {{
            color: #7f8c8d;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Genetic Algorithm Statistical Analysis</h1>
        {results_html}
        <div class="timestamp">
            Generated on: {timestamp}
        </div>
    </div>
</body>
</html>"""

    def _format_significance(self, p_value: float) -> str:
        """Format significance level with HTML styling"""
        if p_value < 0.001:
            return '<span class="significance sig-3">***</span>'
        elif p_value < 0.01:
            return '<span class="significance sig-2">**</span>'
        elif p_value < 0.05:
            return '<span class="significance sig-1">*</span>'
        return '<span class="significance sig-ns">ns</span>'

    def generate_report(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        functions_info: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Generate a complete HTML report with statistical analysis.

        Args:
            results: Dictionary containing experiment results
            functions_info: Dictionary containing function information
        """
        results_html = []

        for func_name in functions_info.keys():
            func_info = functions_info[func_name]
            results_html.append(f'<h2>{func_info["name"]} Results</h2>')

            # 1. Descriptive Statistics Table
            data = []
            config_names = list(results[func_name].keys())

            for config in config_names:
                stats = results[func_name][config]
                data.append(
                    [
                        config,
                        f"{stats['best']:.6f}",
                        f"{stats['mean']:.6f}",
                        f"{stats['std']:.6f}",
                        f"{stats['worst']:.6f}",
                    ]
                )

            df_desc = pd.DataFrame(
                data, columns=["Configuration", "Best", "Mean", "Std Dev", "Worst"]
            )
            results_html.append("<h3>Descriptive Statistics</h3>")
            results_html.append(df_desc.to_html(index=False, classes="table"))

            # 2. ANOVA Test
            results_html.append("<h3>ANOVA Test</h3>")
            groups = [
                results[func_name][config]["fitnesses"] for config in config_names
            ]
            f_stat, p_value = scipy_stats.f_oneway(*groups)
            significance = self._format_significance(p_value)
            results_html.append(f"<p>F-statistic: {f_stat:.6f}</p>")
            results_html.append(f"<p>p-value: {p_value:.6f} {significance}</p>")

            # 3. Pairwise Tests Table
            results_html.append("<h3>Pairwise Statistical Tests</h3>")
            comparisons = []
            for i in range(len(config_names)):
                for j in range(i + 1, len(config_names)):
                    config1, config2 = config_names[i], config_names[j]
                    data1 = results[func_name][config1]["fitnesses"]
                    data2 = results[func_name][config2]["fitnesses"]

                    # t-test
                    t_stat, t_pval = scipy_stats.ttest_ind(data1, data2)
                    t_sig = self._format_significance(t_pval)

                    # Wilcoxon test
                    w_stat, w_pval = scipy_stats.mannwhitneyu(
                        data1, data2, alternative="two-sided"
                    )
                    w_sig = self._format_significance(w_pval)

                    # Effect size (Cohen's d)
                    cohens_d = (np.mean(data1) - np.mean(data2)) / np.sqrt(
                        (np.var(data1) + np.var(data2)) / 2
                    )

                    comparisons.append(
                        [
                            f"{config1} vs {config2}",
                            f"{t_pval:.6f}",
                            t_sig,
                            f"{w_pval:.6f}",
                            w_sig,
                            f"{cohens_d:.3f}",
                        ]
                    )

            df_tests = pd.DataFrame(
                comparisons,
                columns=[
                    "Comparison",
                    "t-test p-value",
                    "t-test sig",
                    "Wilcoxon p-value",
                    "Wilcoxon sig",
                    "Cohen's d",
                ],
            )
            results_html.append(
                df_tests.to_html(index=False, classes="table", escape=False)
            )

            # 4. Summary of Findings
            results_html.append("<h3>Summary of Findings</h3>")
            results_html.append('<div class="summary">')
            if p_value < 0.05:
                results_html.append(
                    "<p>1. ANOVA indicates significant differences between configurations (p < 0.05)</p>"
                )
                results_html.append("<p>2. Pairwise comparisons show:</p><ul>")
                for comp in comparisons:
                    if (
                        comp[2] != '<span class="significance sig-ns">ns</span>'
                        or comp[4] != '<span class="significance sig-ns">ns</span>'
                    ):
                        results_html.append(f"<li>{comp[0]}: Significant difference")
                        results_html.append(
                            f"<br>Effect size (Cohen's d): {comp[5]}</li>"
                        )
                results_html.append("</ul>")
            else:
                results_html.append(
                    "<p>1. ANOVA indicates no significant differences between configurations (p > 0.05)</p>"
                )
                results_html.append(
                    "<p>2. All pairwise comparisons show no significant differences</p>"
                )
            results_html.append("</div>")

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Generate the complete HTML report
        html_content = self.html_template.format(
            results_html="\n".join(results_html),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Save the HTML report
        with open("results/statistical_analysis.html", "w") as f:
            f.write(html_content)

        print(
            "\nStatistical analysis has been saved to 'results/statistical_analysis.html'"
        )
