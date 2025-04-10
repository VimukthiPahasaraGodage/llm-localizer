import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import rankdata
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata
from scipy.stats import wilcoxon
import pickle
import math
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import multilabel_confusion_matrix


class Helper:
    @staticmethod
    def compare_two_models(results, alpha=0.05):
        model_names = sorted(list(results.keys()))

        model_a_results = np.array(results[model_names[0]])
        model_b_results = np.array(results[model_names[1]])

        # Perform Wilcoxon signed-rank test (two-sided by default)
        statistic, p_value = wilcoxon(model_a_results, model_b_results)

        print(f"Wilcoxon statistic = {statistic:.3f}, p-value = {p_value:.4f}")

        if p_value < 0.05:
            print("Statistically significant — meaning one model performs better consistently across folds")
        else:
            print("No significant difference")

        # Determine which model performed better on average
        mean_diff = np.mean(model_a_results - model_b_results)
        print(f"Mean F1 difference ({model_names[0]} - {model_names[1]}): {mean_diff:.4f}")

        if mean_diff > 0:
            print(f"{model_names[0]} is better on average")
        else:
            print(f"{model_names[1]} is better on average")

    @staticmethod
    def compute_critical_difference(k, N, alpha=0.05):
        # Critical values for alpha = 0.05 (from Demšar's table)
        q_alpha_dict = {
            3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
            8: 3.031, 9: 3.102, 10: 3.164
        }
        q_alpha = q_alpha_dict.get(k, 2.569)  # fallback
        cd = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))
        return cd

    @staticmethod
    def plot_cd_diagram_plotly(avg_ranks, model_names, cd, title="Critical Difference Diagram", fig_width=1000,
                               fig_height=400, name="image"):
        avg_ranks = np.array(avg_ranks)
        model_names = np.array(model_names)
        sorted_idx = np.argsort(avg_ranks)
        sorted_ranks = avg_ranks[sorted_idx]
        sorted_names = model_names[sorted_idx]

        fig = go.Figure()
        y = 0
        # Plot model points
        for rank in sorted_ranks:
            fig.add_trace(go.Scatter(
                x=[rank], y=[y],
                mode="markers",
                marker=dict(size=10),
                showlegend=False
            ))

        # Add rotated text labels as annotations
        for rank, name in zip(sorted_ranks, sorted_names):
            fig.add_annotation(
                x=rank,
                y=y + 0.02,  # slight offset above the point
                text=name,
                showarrow=False,
                textangle=-90,  # vertical text (270°)
                font=dict(size=12),
                xanchor="center",
                yanchor="bottom"
            )

        # Plot horizontal axis
        fig.add_shape(type="line", x0=min(sorted_ranks) - 0.2, x1=max(sorted_ranks) + 0.2, y0=y, y1=y,
                      line=dict(color="black", width=1))

        # Plot critical difference line
        x0 = sorted_ranks[-1]
        x1 = x0 - cd
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y - 0.02, y1=y - 0.02,
                      line=dict(color="red", width=3))

        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[y - 0.03],
            mode="text",
            text=[f"CD = {cd:.2f}"],
            showlegend=False
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title="Average Rank", range=[min(avg_ranks) - 0.5, max(avg_ranks) + 0.5]),
            yaxis=dict(visible=False),
            width=fig_width, height=fig_height
        )
        fig.show()
        fig.write_image(f"D:\\results\\{name}.svg")

    @staticmethod
    def friedman_and_nemenyi(f1_scores: dict, fig_width=1000, fig_height=400, name="image"):
        model_names = list(f1_scores.keys())
        score_matrix = np.array([f1_scores[model] for model in model_names]).T  # shape: (n_datasets, n_models)

        # Step 1: Friedman test
        stat, p = friedmanchisquare(*score_matrix.T)
        print(f"\nFriedman Test: statistic = {stat:.4f}, p = {p:.4f}")

        # Step 2: Compute average ranks
        ranks = np.array([rankdata(-row) for row in score_matrix])  # Negative for descending
        avg_ranks = np.mean(ranks, axis=0)

        print("\nAverage Ranks:")
        for name, r in zip(model_names, avg_ranks):
            print(f"{name}: {r:.3f}")

        # Step 3: Nemenyi Post-hoc Test if Friedman is significant
        if p < 0.05:
            df = pd.DataFrame(score_matrix, columns=model_names)
            nemenyi = sp.posthoc_nemenyi_friedman(df)
            print("\nNemenyi Post-hoc Test (p-values):")
            print(pd.DataFrame(nemenyi))

            # Report significant differences
            print("\nSignificant Differences (p < 0.05):")
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    pval = nemenyi.iloc[i, j]
                    if pval < 0.05:
                        winner = model_names[i] if avg_ranks[i] < avg_ranks[j] else model_names[j]
                        loser = model_names[j] if avg_ranks[i] < avg_ranks[j] else model_names[i]
                        print(f"{winner} is significantly better than {loser} (p = {pval:.4f})")

            # Step 4: Critical Difference Diagram
            k = len(model_names)  # number of models
            N = len(score_matrix)  # number of datasets/folds
            cd = Helper.compute_critical_difference(k, N)

            Helper.plot_cd_diagram_plotly(avg_ranks, model_names, cd, fig_width=fig_width, fig_height=fig_height,
                                          name=name)
        else:
            print("\nNo significant difference found (p >= 0.05).")

    @staticmethod
    def plot_grouped_bar_chart(f1_scores_dict: dict, title="F1-scores of Models Across Folds",
                               x_label="Cross-Validation Fold", y_label="F1-score", range_min=0.0, range_max=100.0,
                               fig_height=500, fig_width=1000, name="image"):
        """
        Plots a grouped bar chart of F1-scores for multiple models over multiple folds.

        Parameters:
            f1_scores_dict (dict): A dictionary where keys are model names (str)
                                   and values are lists of F1-scores (float) per fold.
                                   All lists must be of the same length (k folds).
        """
        model_names = list(f1_scores_dict.keys())
        num_folds = len(next(iter(f1_scores_dict.values())))
        folds = [f"Fold {i + 1}" for i in range(num_folds)]

        # Check for consistent number of folds
        for scores in f1_scores_dict.values():
            if len(scores) != num_folds:
                raise ValueError("All models must have the same number of fold scores.")

        # Generate visually distinct colors using Plotly palette
        colors = px.colors.qualitative.Plotly
        if len(model_names) > len(colors):
            colors *= (len(model_names) // len(colors) + 1)  # Repeat palette if needed

        # Create the grouped bar chart
        fig = go.Figure()

        for i, model in enumerate(model_names):
            fig.add_trace(go.Bar(
                x=folds,
                y=f1_scores_dict[model],
                name=model,
                marker_color=colors[i],
                text=[f"{score:.3f}" for score in f1_scores_dict[model]],
                textposition="auto",
                hovertemplate=f"<b>{model}</b><br>Fold: %{{x}}<br>F1-score: %{{y:.3f}}<extra></extra>",
            ))

        # Customize layout for visual appeal
        fig.update_layout(
            title=title,
            xaxis=dict(title=x_label, tickangle=-45),
            yaxis=dict(title=y_label, range=[range_min, range_max]),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.05,
            plot_bgcolor='white',
            legend=dict(title="Models", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Arial", size=14),
            width=fig_width, height=fig_height
        )

        # Add horizontal grid lines
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

        fig.show()
        fig.write_image(f"D:\\results\\{name}.svg")

    @staticmethod
    def plot_boxplots(f1_scores_dict: dict, fig_width=1000, fig_height=400,
                      title="F1-score Distribution Across 10 Folds per Model", x_label="Model", y_label="F1-score",
                      range_min=0.0, range_max=100.00, name="image"):
        """
        Plots a box plot for multiple models showing F1-score distributions.

        Parameters:
            f1_scores_dict (dict): Dictionary where keys are model names (str)
                                   and values are lists of F1-scores (float).
        """
        model_names = list(f1_scores_dict.keys())

        # Generate distinct Plotly colors
        colors = px.colors.qualitative.Set2
        if len(model_names) > len(colors):
            colors *= (len(model_names) // len(colors) + 1)

        # Create the box plot
        fig = go.Figure()

        for i, model in enumerate(model_names):
            fig.add_trace(go.Box(
                y=f1_scores_dict[model],
                name=model,
                marker_color=colors[i],
                boxpoints='outliers',  # Show all points
                jitter=0.5,  # Spread out points for visibility
                pointpos=-1.8,  # Offset points to the left
                marker=dict(size=6, opacity=0.7),
                line=dict(width=2),
                hovertemplate=f"<b>{model}</b><br>F1-score: %{{y:.3f}}<extra></extra>"
            ))

        # Layout styling
        fig.update_layout(
            title=title,
            yaxis_title=y_label,
            xaxis_title=x_label,
            yaxis=dict(range=[range_min, range_max]),
            plot_bgcolor='white',
            font=dict(family="Arial", size=14),
            showlegend=False,
            width=fig_width, height=fig_height
        )

        fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=1)

        fig.show()
        fig.write_image(f"D:\\results\\{name}.svg")

    @staticmethod
    def plot_grouped_boxplots(metrics_data: dict, fig_width=1000, fig_height=400,
                              title="Distribution of Metrics Across Models and Folds", x_label="Model", y_label="Score",
                              range_min=0.0, range_max=40.00, name="image"):
        """
        Creates a grouped box plot for multiple models and multiple metrics with controlled spacing.

        Parameters:
            metrics_data (dict): A nested dictionary of the form:
                {
                    "Model A": {
                        "f1": [...],
                        "precision": [...],
                        "recall": [...],
                    },
                    "Model B": {...},
                    ...
                }
        """
        models = list(metrics_data.keys())
        metrics = list(next(iter(metrics_data.values())).keys())

        # Check data consistency
        for model, metric_dict in metrics_data.items():
            if set(metric_dict.keys()) != set(metrics):
                raise ValueError(f"Model {model} has inconsistent metric keys.")
            lengths = [len(metric_dict[metric]) for metric in metrics]
            if len(set(lengths)) > 1:
                raise ValueError(f"Model {model} has inconsistent fold lengths.")

        # Color palette (loop if needed)
        color_palette = px.colors.qualitative.Set2
        while len(color_palette) < len(metrics):
            color_palette *= 2

        fig = go.Figure()

        # Manual x-axis positions for controlled spacing
        x_positions = []
        x_labels = []
        traces = []

        group_spacing = 1.0  # space between models
        intra_spacing = 0.3  # space within a group

        pos = 0
        for model in models:
            for i, metric in enumerate(metrics):
                x_positions.append(pos)
                x_labels.append(f"{model}<br>{metric}")
                fig.add_trace(go.Box(
                    y=metrics_data[model][metric],
                    x=[pos] * len(metrics_data[model][metric]),  # manual x
                    name=metric,
                    boxpoints='outliers',
                    marker_color=color_palette[i],
                    line=dict(width=2),
                    hovertemplate=f"<b>{model}</b><br>{metric}: %{{y:.3f}}<extra></extra>",
                    showlegend=(model == models[0]),
                    legendgroup=metric,
                ))
                pos += intra_spacing
            pos += (group_spacing - intra_spacing * len(metrics))  # group spacing

        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title=y_label,
            xaxis=dict(
                tickmode='array',
                tickvals=x_positions,
                ticktext=x_labels,
                title=x_label,
            ),
            plot_bgcolor='white',
            font=dict(family="Arial", size=14),
            legend_title_text="Metrics",
            margin=dict(t=80, b=100, l=50, r=50),
            width=fig_width,
            height=fig_height
        )

        fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=1)

        fig.show()
        fig.write_image(f"D:\\results\\{name}.svg")

    @staticmethod
    def plot_precision_recall_curves(
            pr_data,
            summary_method='interpolation',
            template='plotly_white',
            fig_width=1200,
            fig_height=800,
            name="image"
    ):
        """
        Plots precision-recall curves for multiple models and folds using Plotly,
        with per-subplot legends for folds and mean, plus a summary panel legend.

        pr_data: dict { model_name: { fold_idx: {'precision': array, 'recall': array}, ... }, ... }
        summary_method: 'interpolation' (default)
        template: Plotly template name
        fig_width, fig_height: pixel dimensions

        Layout:
          - One subplot per model with internal legend annotation
          - Final subplot comparing models with a built-in legend
        """
        model_names = list(pr_data.keys())
        n_models = len(model_names)
        total_plots = n_models + 1
        cols = int(np.ceil(np.sqrt(total_plots)))
        rows = int(np.ceil(total_plots / cols))

        # color palettes
        fold_palette = px.colors.qualitative.Plotly
        model_palette = px.colors.qualitative.Dark24

        # create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[*model_names, 'Models summary'],
            horizontal_spacing=0.1, vertical_spacing=0.1
        )
        fig.update_layout(
            template=template,
            width=fig_width, height=fig_height,
            title_text='Precision-Recall Curves', title_x=0.5
        )

        summary_curves = {}

        # plot each model's folds and mean
        for idx, model in enumerate(model_names):
            r, c = divmod(idx, cols)
            r += 1;
            c += 1
            folds = pr_data[model]
            recall_grid = np.linspace(0, 1, 200)
            interp_list = []
            legend_html = ''

            # folds
            for i, (fold_idx, data) in enumerate(folds.items()):
                color = fold_palette[i % len(fold_palette)]
                rec = np.array(data['recall']);
                prec = np.array(data['precision'])
                order = np.argsort(rec)
                rec_s, prec_s = rec[order], prec[order]
                interp_list.append(np.interp(recall_grid, rec_s, prec_s))
                fig.add_trace(
                    go.Scatter(
                        x=rec_s, y=prec_s,
                        mode='lines', line=dict(color=color, width=1),
                        name=f'Fold {fold_idx}', showlegend=False,
                        hovertemplate=f'Fold {fold_idx}<br>Recall: %{{x:.2f}}<br>Precision: %{{y:.2f}}'
                    ), row=r, col=c
                )
                legend_html += f"<span style='color:{color};'>───</span> Fold {fold_idx}<br>"

            # mean
            mean_prec = np.mean(interp_list, axis=0)
            summary_curves[model] = (recall_grid, mean_prec)
            fig.add_trace(
                go.Scatter(
                    x=recall_grid, y=mean_prec,
                    mode='lines', line=dict(color='black', width=3),
                    name='Mean', showlegend=False,
                    hovertemplate='Mean<br>Recall: %{x:.2f}<br>Precision: %{y:.2f}'
                ), row=r, col=c
            )
            legend_html += "<span style='color:black;'>───</span> Mean"

            # internal legend
            fig.add_annotation(
                text=legend_html,
                xref='x domain', yref='y domain',
                x=0.02, y=0.98,
                showarrow=False, align='left',
                font=dict(size=10), bgcolor='rgba(255,255,255,0.8)',
                row=r, col=c
            )

        # summary panel
        idx = n_models
        r, c = divmod(idx, cols)
        r += 1;
        c += 1
        for mi, model in enumerate(model_names):
            color = model_palette[mi % len(model_palette)]
            rec, prec = summary_curves[model]
            fig.add_trace(
                go.Scatter(
                    x=rec, y=prec,
                    mode='lines', line=dict(color=color, width=3),
                    name=model, showlegend=True,
                    hovertemplate=f'{model}<br>Recall: %{{x:.2f}}<br>Precision: %{{y:.2f}}'
                ), row=r, col=c
            )

        # enable and style global legend
        fig.update_layout(
            showlegend=True,
            legend=dict(
                x=1.02, y=1, traceorder='normal',
                font=dict(size=10), bgcolor='rgba(255,255,255,0.8)'
            )
        )

        # axes
        for i in range(total_plots):
            rr, cc = divmod(i, cols)
            fig.update_xaxes(title_text='Recall', row=rr + 1, col=cc + 1)
            fig.update_yaxes(title_text='Precision', row=rr + 1, col=cc + 1)

        fig.show()
        fig.write_image(f"D:\\results\\{name}.svg")

    @staticmethod
    def extract_results(results_folder, experiment):
        # Load the CSV file
        df = pd.read_csv(f"{results_folder}/{experiment.split('/')[-1]}.csv")  # Replace with your actual CSV file path

        # Set 'metric' column as the index
        df.set_index('metric', inplace=True)

        # Convert each metric row to a list of fold values
        metrics_dict = {metric: df.loc[metric].tolist() for metric in df.index}
        return metrics_dict

    @staticmethod
    def plot_all_folds_confusion(outputs_outputs_folder, outputs_csv_folder, exp, classes):
        def get_results_for_confusion_matrix(outputs_outputs_folder, outputs_csv_folder, exp):
            def sigmoid_classify_list(x_list):
                results = []
                for x in x_list:
                    sigmoid = 1 / (1 + math.exp(-x))
                    classification = 1 if sigmoid >= 0.5 else 0
                    results.append(classification)
                return results

            results = {}
            for fold in range(10):
                try:
                    with open(f'{os.getcwd()}/{outputs_outputs_folder}/{exp}/outputs/fold_{fold}/val_outputs.pkl',
                              'rb') as f:
                        data = pickle.load(f)

                    epochs = list(data.keys())
                    if len(epochs) > 0:
                        results[fold] = {}
                        df = pd.read_csv(f'{os.getcwd()}/{outputs_csv_folder}/{exp}/fold_{fold}/f1_score_val.csv')
                        sorted_steps = df.sort_values(by='value', ascending=False)['step'].tolist()
                        for step in sorted_steps:
                            if step in epochs:
                                preds = []
                                labels = []
                                preds_and_labels = data[step]
                                for item in preds_and_labels:
                                    preds.append(list(sigmoid_classify_list(item['outputs'])))
                                    labels.append(item['labels'])
                                results[fold]['preds'] = preds
                                results[fold]['labels'] = labels
                                break
                except Exception as e:
                    print(f"Fold {fold} failed to load: {e}")
            return results

        results = get_results_for_confusion_matrix(outputs_outputs_folder, outputs_csv_folder, exp)

        num_folds = len(results)
        num_classes = len(classes)

        fig = make_subplots(
            rows=num_folds,
            cols=num_classes,
            subplot_titles=[f"Fold {f} - Class {c}" for f in results for c in classes],
            horizontal_spacing=0.03,
            vertical_spacing=0.05
        )

        for row_idx, (fold, data) in enumerate(results.items(), start=1):
            y_true = np.array(data['labels'])
            y_pred = np.array(data['preds'])

            mcm = multilabel_confusion_matrix(y_true, y_pred)

            for col_idx, (cls, matrix) in enumerate(zip(classes, mcm), start=1):
                tn, fp, fn, tp = matrix.ravel()
                z = [[tp, fn], [fp, tn]]
                text = [[f"{tp}", f"{fn}"], [f"{fp}", f"{tn}"]]

                heatmap = go.Heatmap(
                    z=z,
                    x=["Pred 1", "Pred 0"],
                    y=["True 1", "True 0"],
                    text=text,
                    texttemplate="%{text}",
                    hoverinfo='text',
                    colorscale='Blues',
                    showscale=False
                )

                fig.add_trace(heatmap, row=row_idx, col=col_idx)

        fig.update_layout(
            title="Confusion Matrices by Fold and Class",
            height=300 * num_folds,
            width=300 * num_classes,
            margin=dict(t=50),
            font=dict(size=10)
        )

        fig.show()


outputs_folder = "new_outputs"
outputs_csv_folder = "new_tensorboard_csvs"
outputs_result_comparison_folder = "new_results_comparison"

experiments = ['defects4j/v2/codegen_350M/exp1_defects4j_v2_1_linear_1024_1_BCE',
               'defects4j/v2/codegen_6B/exp1_defects4j_v2_2_linear_1024_1_BCE',
               'defects4j/v2/codegen_16B/exp1_defects4j_v2_3_linear_1024_1_BCE',
               'defects4j/v2/codegen_16B/exp2_defects4j_v2_3_gru_1024_1_BCE',
               'defects4j/v2/Qwen_QwQ_32B/exp3_defects4j_v2_17_linear_1024_1_BCE',
               'defects4j/v2/DeepSeek_R1_Distill_Llama_8B/exp4_defects4j_v2_18_linear_1024_1_BCE',
               'defects4j/v2/codegen_16B/exp5_defects4j_v2_3_linear_1024_1_CC',

               'solidity/v2/Qwen_QwQ_32B/exp6_solidity_v2_17_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Llama_8B/exp7_solidity_v2_18_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp8_solidity_v2_19_gru_1024_1_BCE',
               'solidity/v3/codegen_16B/exp9_solidity_v3_3_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp10_solidity_v2_19_gru_1024_2_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp11_solidity_v2_19_gru_1024_2_BCE']
experiment_results = dict()
for experiment in experiments:
    experiment_results[experiment] = Helper.extract_results(outputs_result_comparison_folder, experiment)


experiments = ['solidity/v2/Qwen_QwQ_32B/exp6_solidity_v2_17_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Llama_8B/exp7_solidity_v2_18_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp8_solidity_v2_19_gru_1024_1_BCE',
               'solidity/v3/codegen_16B/exp9_solidity_v3_3_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp10_solidity_v2_19_gru_1024_2_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp11_solidity_v2_19_gru_1024_2_BCE']

Helper.friedman_and_nemenyi({'Qwen-QwQ-32B + GRU - Configuration 1': experiment_results['solidity/v2/Qwen_QwQ_32B/exp6_solidity_v2_17_gru_1024_1_BCE']['f1_score'],
                             'DeepSeek-R1-Distill-Llama-8B + GRU - Configuration 1': experiment_results['solidity/v2/DeepSeek_R1_Distill_Llama_8B/exp7_solidity_v2_18_gru_1024_1_BCE']['f1_score'],
                             'DeepSeek-R1-Distill-Qwen-14B + GRU - Configuration 1': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp8_solidity_v2_19_gru_1024_1_BCE']['f1_score'],
                             'CodeGen-16B + GRU  - Configuration 1': experiment_results['solidity/v3/codegen_16B/exp9_solidity_v3_3_gru_1024_1_BCE']['f1_score'],
                             'DeepSeek-R1-Distill-Qwen-14B + GRU - Configuration 2': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp10_solidity_v2_19_gru_1024_2_BCE']['f1_score'],
                             'DeepSeek-R1-Distill-Qwen-14B + GRU - Configuration 2 (No prompt)': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp11_solidity_v2_19_gru_1024_2_BCE']['f1_score']},
                            fig_width=1000, fig_height=800, name="localization_cd")

Helper.plot_boxplots({'Qwen-QwQ-32B + GRU - Configuration 1': experiment_results['solidity/v2/Qwen_QwQ_32B/exp6_solidity_v2_17_gru_1024_1_BCE']['f1_score'],
                             'DeepSeek-R1-Distill-Llama-8B + GRU - Configuration 1': experiment_results['solidity/v2/DeepSeek_R1_Distill_Llama_8B/exp7_solidity_v2_18_gru_1024_1_BCE']['f1_score'],
                             'DeepSeek-R1-Distill-Qwen-14B + GRU - Configuration 1': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp8_solidity_v2_19_gru_1024_1_BCE']['f1_score'],
                             'CodeGen-16B + GRU  - Configuration 1': experiment_results['solidity/v3/codegen_16B/exp9_solidity_v3_3_gru_1024_1_BCE']['f1_score'],
                             'DeepSeek-R1-Distill-Qwen-14B + GRU - Configuration 2': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp10_solidity_v2_19_gru_1024_2_BCE']['f1_score'],
                             'DeepSeek-R1-Distill-Qwen-14B + GRU - Configuration 2 (No prompt)': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp11_solidity_v2_19_gru_1024_2_BCE']['f1_score']},
                    fig_width=1000, fig_height=1000, title="F1-score Distribution Across 10 Folds per Model", x_label="Model", y_label="F1-score", range_min=50.00, range_max=90.00, name="localization_f1_box")


Helper.plot_grouped_boxplots({'Qwen-QwQ-32B + GRU + BCE': {'Top 1': experiment_results['solidity/v2/Qwen_QwQ_32B/exp6_solidity_v2_17_gru_1024_1_BCE']['top_1'],
                                               'Top 3': experiment_results['solidity/v2/Qwen_QwQ_32B/exp6_solidity_v2_17_gru_1024_1_BCE']['top_3'],
                                               'Top 5': experiment_results['solidity/v2/Qwen_QwQ_32B/exp6_solidity_v2_17_gru_1024_1_BCE']['top_5']},
                      'DeepSeek-R1-Distill-Llama-8B + GRU + BCE': {'Top 1': experiment_results['solidity/v2/DeepSeek_R1_Distill_Llama_8B/exp7_solidity_v2_18_gru_1024_1_BCE']['top_1'],
                                     'Top 3': experiment_results['solidity/v2/DeepSeek_R1_Distill_Llama_8B/exp7_solidity_v2_18_gru_1024_1_BCE']['top_3'],
                                     'Top 5': experiment_results['solidity/v2/DeepSeek_R1_Distill_Llama_8B/exp7_solidity_v2_18_gru_1024_1_BCE']['top_5']},
                      'DeepSeek-R1-Distill-Qwen-14B + GRU + BCE': {'Top 1': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp8_solidity_v2_19_gru_1024_1_BCE']['top_1'],
                                      'Top 3': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp8_solidity_v2_19_gru_1024_1_BCE']['top_3'],
                                      'Top 5': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp8_solidity_v2_19_gru_1024_1_BCE']['top_5']},
                             'CodeGen-16B + GRU + BCE': {'Top 1': experiment_results['solidity/v3/codegen_16B/exp9_solidity_v3_3_gru_1024_1_BCE']['top_1'],
                                               'Top 3': experiment_results['solidity/v3/codegen_16B/exp9_solidity_v3_3_gru_1024_1_BCE']['top_3'],
                                               'Top 5': experiment_results['solidity/v3/codegen_16B/exp9_solidity_v3_3_gru_1024_1_BCE']['top_5']},
                             'DeepSeek-R1-Distill-Qwen-14B + GRU + Config 1 + BCE': {'Top 1': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp10_solidity_v2_19_gru_1024_2_BCE']['top_1'],
                                      'Top 3': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp10_solidity_v2_19_gru_1024_2_BCE']['top_3'],
                                      'Top 5': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp10_solidity_v2_19_gru_1024_2_BCE']['top_5']},
                             'DeepSeek-R1-Distill-Qwen-14B + GRU + Config 2 + BCE (No prompt)': {'Top 1': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp11_solidity_v2_19_gru_1024_2_BCE']['top_1'],
                                      'Top 3': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp11_solidity_v2_19_gru_1024_2_BCE']['top_3'],
                                      'Top 5': experiment_results['solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp11_solidity_v2_19_gru_1024_2_BCE']['top_5']}},
                    fig_width=1000, fig_height=800, title="Distribution of Top 1, Top 3 and Top 5 Metrics Across Models and Folds", x_label="Model and Metric", y_label="Score", range_min=0.0, range_max=100.00, name="localization_top_box")

# pr_data = { model_name: { fold_idx: {'precision': array, 'recall': array}, ... }, ... }
import pandas as pd
min_steps = 10000000
sub_experiments = ['solidity/v2/Qwen_QwQ_32B/exp6_solidity_v2_17_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Llama_8B/exp7_solidity_v2_18_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp8_solidity_v2_19_gru_1024_1_BCE',
               'solidity/v3/codegen_16B/exp9_solidity_v3_3_gru_1024_1_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp10_solidity_v2_19_gru_1024_2_BCE',
               'solidity/v2/DeepSeek_R1_Distill_Qwen_14B/exp11_solidity_v2_19_gru_1024_2_BCE']
for experiment in sub_experiments:
    for k in range(10):
        val_precision_file = f"{outputs_csv_folder}/{experiment}/fold_{k}/precision_val.csv"
        val_recall_file = f"{outputs_csv_folder}/{experiment}/fold_{k}/recall_val.csv"
        val_precision_df = pd.read_csv(val_precision_file)
        val_recall_df = pd.read_csv(val_recall_file)
        val_precision = val_precision_df['value'].tolist()
        val_recall = val_recall_df['value'].tolist()
        min_steps_temp = min(len(val_precision), len(val_recall))
        min_steps = min(min_steps, min_steps_temp)

pr_data = dict()
for experiment in sub_experiments:
    model_name = experiment.split('/')[2].replace("_", "-")
    if 'linear' in experiment.split('/')[3]:
        model_name += " + Linear"
    if 'gru' in experiment.split('/')[3]:
        model_name += " + GRU"
    if '1' in experiment.split('/')[3].split('_')[6]:
        model_name += " + Confi 1"
    if '2' in experiment.split('/')[3].split('_')[6]:
        model_name += " + Config 2"
    if 'BCE' in experiment.split('/')[3]:
        model_name += " + BCE"
    if 'CC' in experiment.split('/')[3]:
        model_name += " + Custom Loss"
    if 'exp11' in experiment.split('/')[3]:
        model_name += " (No prompt)"
    if 'exp10' in experiment.split('/')[3]:
        model_name += " (With prompt)"
    pr_data[model_name] = dict()
    for k in range(10):
        val_precision_file = f"{outputs_csv_folder}/{experiment}/fold_{k}/precision_val.csv"
        val_recall_file = f"{outputs_csv_folder}/{experiment}/fold_{k}/recall_val.csv"
        val_precision_df = pd.read_csv(val_precision_file)
        val_recall_df = pd.read_csv(val_recall_file)
        val_precision = val_precision_df['value'].tolist()
        val_recall = val_recall_df['value'].tolist()
        pr_data[model_name][k] = {'precision': val_precision[:min_steps], 'recall': val_recall[:min_steps]}

#print(pr_data)

Helper.plot_precision_recall_curves(
        pr_data,
        summary_method='interpolation',
        template='plotly_white',
        fig_width=5000,
        fig_height=5000,
        name="localization_f1_box"
    )

Helper.plot_grouped_boxplots({'binary classification': {'Config 1': experiment_results['solidity_detect_1/v1/DeepSeek_R1_Distill_Qwen_14B/exp_detection_solidity_detect_1_v1_19_1_BCE']['f1_score'],
                                               'Config 2': experiment_results['solidity_detect_1/v1/DeepSeek_R1_Distill_Qwen_14B/exp_detection_solidity_detect_1_v1_19_2_BCE']['f1_score']},
                      '3 classes classification': {'Config 1': experiment_results['solidity_detect_3/v1/DeepSeek_R1_Distill_Qwen_14B/exp_detection_solidity_detect_3_v1_19_1_BCE']['f1_score'],
                                     'Config 2': experiment_results['solidity_detect_3/v1/DeepSeek_R1_Distill_Qwen_14B/exp_detection_solidity_detect_3_v1_19_2_BCE']['f1_score']},
                      '15 classes classification': {'Config 1': experiment_results['solidity_detect_15/v1/DeepSeek_R1_Distill_Qwen_14B/exp_detection_solidity_detect_15_v1_19_1_BCE']['f1_score'],
                                      'Config 2': experiment_results['solidity_detect_15/v1/DeepSeek_R1_Distill_Qwen_14B/exp_detection_solidity_detect_15_v1_19_2_BCE']['f1_score']}},
                    fig_width=1000, fig_height=1000, title="Distribution of F1-Scores Across Models & Configurations and Folds", x_label="Model Configurations", y_label="F1 Score", range_min=0.0, range_max=100.00, name="detection_f1")