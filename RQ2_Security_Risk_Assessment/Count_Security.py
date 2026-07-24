import pandas as pd
import os

def calculate_risk_exposure():
    report_file = 'Widely_used_deprecated.csv'
    vuln_file = 'Non_GDNPs_assessment.csv'

    if not os.path.exists(report_file) or not os.path.exists(vuln_file):
        print("error: Input files do not exist.")
        return

    try:
        df_report = pd.read_csv(report_file, on_bad_lines='skip')
        df_vuln = pd.read_csv(vuln_file, on_bad_lines='skip')

        df_report.columns = df_report.columns.str.strip()
        df_vuln.columns = df_vuln.columns.str.strip()

        required_cols_report = ['package_name', 'last_month_downloads']
        required_cols_vuln = ['package_name', 'severity_rating']
        
        if not all(col in df_report.columns for col in required_cols_report):
            return
        if not all(col in df_vuln.columns for col in required_cols_vuln):
            return

        merged_df = pd.merge(df_report, df_vuln, on='package_name', how='inner')

        if merged_df.empty:
            print("error: No matching package data found.")
            return

        total_all = merged_df['last_month_downloads'].sum()
        total_by_severity = merged_df.groupby('severity_rating')['last_month_downloads'].sum()

        unique_packages_all = merged_df['package_name'].nunique()
        mean_all = total_all / unique_packages_all if unique_packages_all > 0 else 0

        severity_package_count = merged_df.drop_duplicates(subset=['package_name', 'severity_rating']).groupby('severity_rating').size()
        mean_by_severity = (total_by_severity / severity_package_count).fillna(0)

        package_total_exposure = merged_df.groupby('package_name')['last_month_downloads'].sum()
        median_all = package_total_exposure.median()

        severity_package_exposure = merged_df.groupby(['package_name', 'severity_rating'])['last_month_downloads'].sum().reset_index()
        median_by_severity = severity_package_exposure.groupby('severity_rating')['last_month_downloads'].median()

        print("--- (Total) ---")
        print(f"All vulnerabilities: {total_all}")
        for severity, total_val in total_by_severity.items():
            print(f"{severity}: {total_val}")

        print("\n--- (Mean) ---")
        print(f"Average risk exposure per package for each vulnerability type: {mean_all:.2f}")
        for severity, mean_val in mean_by_severity.items():
            print(f"{severity}: {mean_val:.2f}")

        print("\n--- (Median) ---")
        print(f"Median risk exposure per package for each vulnerability type: {median_all:.2f}")
        for severity, median_val in median_by_severity.items():
            print(f"{severity}: {median_val:.2f}")

    except Exception as e:
        print(f"error: An error occurred while running the program: {e}")

if __name__ == "__main__":
    calculate_risk_exposure()