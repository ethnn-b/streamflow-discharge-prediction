import re

with open("src/experiments/benchmark_1980_1990.py", "r") as f:
    text = f.read()

pattern = re.compile(r"def plot_selected_gaps\(.*?\n\n\ndef run_benchmark", re.DOTALL)

replacement = """def plot_selected_gaps_subplots(df_original, data_list, discharge_cols, out_dir, method_name):
    print(f"    Plotting best gaps subplots for {method_name}...")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        import matplotlib.pyplot as sns
    import os
    import numpy as np
    import pandas as pd
    
    completeness = df_original[discharge_cols].notna().sum()
    target_station = completeness.idxmax()
    
    n_gaps = len(data_list)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})
    
    fig, axes = plt.subplots(nrows=n_gaps, ncols=1, figsize=(12, 4 * n_gaps), squeeze=False)
    
    for i, (gap_length, df_gapped, df_imputed) in enumerate(data_list):
        ax = axes[i, 0]
        
        gap_candidates = []
        gap_mask = df_gapped[target_station].isnull() & df_original[target_station].notnull()
        is_gap = gap_mask.astype(int)
        gap_starts = is_gap[(is_gap == 1) & (is_gap.shift(1, fill_value=0) == 0)].index
        gap_ends = is_gap[(is_gap == 1) & (is_gap.shift(-1, fill_value=0) == 0)].index
        
        for start_idx, end_idx in zip(gap_starts, gap_ends):
            orig_vals = df_original.loc[start_idx:end_idx, target_station]
            imp_vals = df_imputed.loc[start_idx:end_idx, target_station]
            if len(orig_vals) >= 3: 
                variance = orig_vals.var()
                if pd.notna(variance):
                    mse = ((orig_vals - imp_vals) ** 2).mean()
                    if pd.notna(mse):
                        gap_candidates.append((start_idx, end_idx, variance, mse))
                    
        if gap_candidates:
            variances = [g[2] for g in gap_candidates]
            var_threshold = np.percentile(variances, 75)
            high_var_gaps = [g for g in gap_candidates if g[2] >= var_threshold]
            if not high_var_gaps:
                high_var_gaps = gap_candidates
            best_gap = min(high_var_gaps, key=lambda g: g[3])
            station, start_idx, end_idx = target_station, best_gap[0], best_gap[1]
            
            context_days = min(15, max(5, gap_length // 2))
            plot_start = max(start_idx - pd.Timedelta(days=context_days), df_original.index.min())
            plot_end = min(end_idx + pd.Timedelta(days=context_days), df_original.index.max())
            
            ax.plot(df_original.loc[plot_start:plot_end].index, df_original.loc[plot_start:plot_end, station], 
                    label='Original', color='#5B9BD5', linewidth=2.5, alpha=0.8)
            ax.plot(df_imputed.loc[start_idx:end_idx].index, df_imputed.loc[start_idx:end_idx, station], 
                    label=f'Imputed', color='#FFC000', linewidth=2.5, linestyle='--')
            ax.axvspan(start_idx, end_idx, color='gray', alpha=0.15, label='Gap Region')
            
            ax.set_title(f'Gap: {gap_length} days', pad=5)
            ax.set_ylabel('Discharge')
            
            ax.yaxis.grid(True, linestyle='-', linewidth=1, color='#D9D9D9')
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            sns.despine(ax=ax, left=True, bottom=False, top=True, right=True)
            ax.spines['bottom'].set_color('#D9D9D9')
            if i == 0:
                ax.legend(loc='upper right', frameon=True)
                
    fig.suptitle(f'Zoomed Hydrographs - Station: {target_station} ({method_name})', y=1.02, fontsize=16)
    plt.tight_layout()
    plot_filename = os.path.join(out_dir, f"zoomed_subplots_{method_name.replace(' ', '_')}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved zoomed subplots to {plot_filename}")


def plot_overall_results_subplots(df_original, data_list, discharge_cols, out_dir, method_name):
    print(f"    Plotting overall subplots for {method_name}...")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        import matplotlib.pyplot as sns
    import numpy as np
    import pandas as pd
    import os
    
    completeness = df_original[discharge_cols].notna().sum()
    target_station = completeness.idxmax()
    n_gaps = len(data_list)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})
    
    # 2 columns: left is Hydrograph, right is R^2 scatter
    fig, axes = plt.subplots(nrows=n_gaps, ncols=2, figsize=(20, 5 * n_gaps), squeeze=False)
    
    for i, (gap_length, df_gapped, df_imputed) in enumerate(data_list):
        ax_hydro = axes[i, 0]
        ax_scatter = axes[i, 1]
        
        # --- Scatter Plot (Global) ---
        all_true_vals = []
        all_pred_vals = []
        for station in discharge_cols:
            gap_mask = df_gapped[station].isnull() & df_original[station].notnull()
            if gap_mask.sum() > 0:
                all_true_vals.extend(df_original.loc[gap_mask, station].values)
                all_pred_vals.extend(df_imputed.loc[gap_mask, station].values)
                
        true_vals_global = pd.Series(all_true_vals)
        pred_vals_global = pd.Series(all_pred_vals)
        
        if not true_vals_global.empty:
            ax_scatter.scatter(true_vals_global, pred_vals_global, alpha=0.3, edgecolors='k', label=f'Imputed (Gaps: {gap_length}d)')
            min_val = min(true_vals_global.min(), pred_vals_global.min())
            max_val = max(true_vals_global.max(), pred_vals_global.max())
            
            if len(true_vals_global) > 1:
                ss_tot = np.sum((true_vals_global - np.mean(true_vals_global))**2)
                ss_res = np.sum((true_vals_global - pred_vals_global)**2)
                r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                r2_val = np.nan
                
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label=f'1:1 Line ($R^2={r2_val:.3f}$)')
            ax_scatter.set_title(f'Scatter ({gap_length}d)')
            ax_scatter.set_xlabel('True Values')
            ax_scatter.set_ylabel('Predicted Values')
            ax_scatter.grid(True, linestyle='-', linewidth=1, color='#D9D9D9')
            ax_scatter.legend(loc='upper left')

        # --- Hydrograph (Target Station) ---
        target_mask = df_gapped[target_station].isnull() & df_original[target_station].notnull()
        pred_vals_station = df_imputed.loc[target_mask, target_station]
        
        ax_hydro.plot(df_original.index, df_original[target_station], label='Original Data', color='#5B9BD5', linewidth=1.5, alpha=0.9)
        if not pred_vals_station.empty:
            ax_hydro.scatter(pred_vals_station.index, pred_vals_station.values, color='red', label=f'Imputed ({gap_length}d)', zorder=5)
        
        ax_hydro.set_title(f'Hydrograph ({gap_length}d)')
        ax_hydro.set_ylabel('Discharge')
        
        ax_hydro.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#D9D9D9')
        ax_hydro.xaxis.grid(True, linestyle='--', linewidth=0.5, color='#D9D9D9')
        ax_hydro.set_axisbelow(True)
        sns.despine(ax=ax_hydro, left=True, bottom=False, top=True, right=True)
        ax_hydro.spines['bottom'].set_color('#D9D9D9')
        if i == 0:
            ax_hydro.legend(loc='upper right', frameon=True)
            
    fig.suptitle(f'Overall Hydrographs and Scatter - Station: {target_station} ({method_name})', y=1.02, fontsize=16)
    plt.tight_layout()
    overall_filename = os.path.join(out_dir, f"overall_subplots_{method_name.replace(' ', '_')}.png")
    plt.savefig(overall_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved overall subplots to {overall_filename}")


def run_benchmark"""

new_text = pattern.sub(replacement, text)

with open("src/experiments/benchmark_1980_1990.py", "w") as f:
    f.write(new_text)

print(len(text), "->", len(new_text))
