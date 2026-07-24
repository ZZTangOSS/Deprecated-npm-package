import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

file_path = 'GDNPs.csv'

def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    if 'slope' not in df.columns:
        raise ValueError("CSV file is missing the 'slope' column")
        
    df['slope'] = pd.to_numeric(df['slope'], errors='coerce')
    clean_df = df.dropna(subset=['slope'])
    
    clean_df = clean_df[
        (clean_df['slope'] > 0) & 
        (np.isfinite(clean_df['slope']))
    ].copy()
    return clean_df

def plot_slope_distribution_scientific(file_path):
    plt.rcParams['text.usetex'] = True
    sns.set_theme(style="ticks", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False 

    try:
        clean_df = load_and_clean_data(file_path)
        if clean_df.empty: return

        clean_df['growth_pct'] = (np.exp(clean_df['slope']) - 1) * 100
        
        median_val = clean_df['growth_pct'].median()
        mean_val = clean_df['growth_pct'].mean()
        max_val = clean_df['growth_pct'].max()
        n_samples = len(clean_df)
        
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        x_display_max = clean_df['growth_pct'].quantile(0.98) 
        bins_main = np.linspace(0, x_display_max, 40) 

        ax.hist(clean_df['growth_pct'], bins=bins_main, color="#4C72B0", alpha=0.7, edgecolor="white")
        ax.set_xlim(0, x_display_max)
        
        ax.axvline(median_val, color='#C44E52', linestyle='--', linewidth=2, 
                   label=r'Median ($\tilde{x}$): ' + f'{median_val:.1f}\%') 
        ax.axvline(mean_val, color='#55A868', linestyle='-', linewidth=2, 
                   label=r'Mean ($\mu$): ' + f'{mean_val:.1f}\%') 

        legend = ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                           title=r'$N$: ' + f'{n_samples:,.0f}', 
                           frameon=True, fancybox=True, framealpha=0.9, 
                           edgecolor='gray', fontsize=11)
        legend.get_title().set_fontsize(11)
        legend.get_title().set_fontweight('bold')

        ax.set_xlabel(r"Monthly Percentage Growth Rate (\%) [Core $98\%$]", fontweight='bold')
        ax.set_ylabel(r"Count of Packages", fontweight='bold')
        ax.grid(True, which="major", axis="y", ls="--", alpha=0.5) 
        ax.grid(True, which="major", axis="x", ls="-", alpha=0.2) 
        
        ax_inset = fig.add_axes([0.52, 0.25, 0.40, 0.30]) 
        bins_inset = np.linspace(0, max_val, 50)
        ax_inset.hist(clean_df['growth_pct'], bins=bins_inset, color="#4C72B0", alpha=0.8, edgecolor="none")
        ax_inset.set_yscale('log')
        ax_inset.set_xlim(0, max_val * 1.05)
        ax_inset.set_title(r"\textit{Full Range}", fontsize=10, pad=5)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        ax_inset.grid(True, ls=':', alpha=0.4)
        
        sns.despine(ax=ax)
        plt.savefig('growth_rate_final.png', bbox_inches='tight', dpi=300)
        plt.savefig('growth_rate_final.pdf', bbox_inches='tight')
        print("Images saved as 'growth_rate_final.png' and 'growth_rate_final.pdf'")
        
    except Exception as e:
        print(f"Error occurred while plotting: {e}")

if __name__ == "__main__":
    plot_slope_distribution_scientific(file_path)