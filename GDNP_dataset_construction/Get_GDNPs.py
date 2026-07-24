import pandas as pd
import requests
import time
import logging
import argparse
import os
import json
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dateutil.parser import isoparse
import numpy as np
import warnings
from typing import Dict, Optional, Tuple, List, Any
from tqdm import tqdm
import concurrent.futures
from dataclasses import dataclass
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class AnalysisResult:
    package_name: str
    deprecation_date: str
    deprecation_message: str
    slope: float
    p_value: float
    r_squared: float
    npm_url: str

NPM_REGISTRY_URL = "https://registry.npmjs.org/{package_name}"
NPM_DOWNLOADS_URL = "https://api.npmjs.org/downloads/range/{start_date}:{end_date}/{package_name}"
CACHE_DIR = ".cache"
FAILED_LOG_FILE = "failed_packages.log"

warnings.filterwarnings('ignore', category=FutureWarning)

def sanitize_filename(name: str) -> str:
    return name.replace('/', '__').replace('@', '')

def requests_retry_session(retries=3, backoff_factor=0.5, session=None):
    session = session or requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.packages.urllib3.util.retry.Retry(
            total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
        )
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def get_from_cache(key: str) -> Optional[Any]:
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def save_to_cache(key: str, data: Any):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except IOError:
        tqdm.write(f"  [!] Failed to write cache: {cache_file}")

def get_deprecation_info(session: requests.Session, package_name: str, use_cache: bool) -> Tuple[Optional[str], Optional[str]]:
    sanitized_name = sanitize_filename(package_name)
    cache_key = f"meta_{sanitized_name}"
    if use_cache:
        cached_data = get_from_cache(cache_key)
        if cached_data: return cached_data.get('msg'), cached_data.get('ts')

    try:
        url = NPM_REGISTRY_URL.format(package_name=package_name)
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        time_data = data.get('time', {})
        t_modified_str = time_data.get('modified')
        if not t_modified_str: return "Metadata missing 'modified' field", None
        
        latest_deprecated_version_date = None
        deprecation_message = None

        for version, publish_time_str in sorted(time_data.items(), key=lambda item: item[1], reverse=True):
            if version in ['created', 'modified']: continue
            version_info = data.get('versions', {}).get(version, {})
            if 'deprecated' in version_info:
                deprecation_message = version_info['deprecated']
                latest_deprecated_version_date = isoparse(publish_time_str)
                break
        
        if not deprecation_message or not latest_deprecated_version_date:
            return None, None
        
        t_modified = isoparse(t_modified_str)
        true_deprecation_date = max(latest_deprecated_version_date, t_modified)
        
        result_ts = true_deprecation_date.isoformat()
        if use_cache:
            save_to_cache(cache_key, {'msg': deprecation_message, 'ts': result_ts})
        
        return deprecation_message, result_ts

    except requests.exceptions.RequestException as e:
        return f"Network Error: {str(e)[:100]}", None
    except (KeyError, ValueError, TypeError) as e:
        return f"Data Parsing Error: {e}", None

def get_monthly_downloads(session: requests.Session, package_name: str, start_date: str, end_date: str, use_cache: bool) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    sanitized_name = sanitize_filename(package_name)
    cache_key = f"dl_{sanitized_name}_{start_date}_{end_date}"
    if use_cache:
        cached_data = get_from_cache(cache_key)
        if cached_data: return cached_data, None

    try:
        url = NPM_DOWNLOADS_URL.format(start_date=start_date, end_date=end_date, package_name=package_name)
        response = session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        downloads_data = data.get('downloads', [])
        if not downloads_data: 
            return None, None
        monthly_downloads: Dict[str, int] = {}
        for daily_record in downloads_data:
            day, downloads = daily_record.get('day'), daily_record.get('downloads')
            if day and downloads is not None:
                month_key = day[:7]
                monthly_downloads[month_key] = monthly_downloads.get(month_key, 0) + downloads
        
        if use_cache:
            save_to_cache(cache_key, monthly_downloads)
        return monthly_downloads, None
    except requests.exceptions.RequestException as e:
        return None, f"Network Error fetching downloads: {str(e)[:100]}"
    except (KeyError, ValueError) as e:
        return None, f"Parsing Error fetching downloads: {e}"

def analyze_trend(monthly_downloads: Dict[str, int], iqr_factor: float) -> Optional[Tuple[float, float, float]]:
    if not monthly_downloads:
        return None
        
    sorted_months = sorted(monthly_downloads.keys())
    downloads = np.array([monthly_downloads[month] for month in sorted_months])
    time_indices = np.arange(len(downloads))

    q1 = np.percentile(downloads, 25)
    q3 = np.percentile(downloads, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    
    non_outlier_mask = (downloads >= lower_bound) & (downloads <= upper_bound)
    
    if np.sum(non_outlier_mask) < 3: 
        return None
        
    y_raw = downloads[non_outlier_mask]
    x_filtered = time_indices[non_outlier_mask]

    y_log = np.log1p(y_raw)
    X = sm.add_constant(x_filtered)

    try:
        model = sm.OLS(y_log, X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        
        slope = results.params[1]
        p_value = results.pvalues[1]
        r_squared = results.rsquared
        
        return slope, p_value, r_squared
    except Exception as e:
        logging.debug(f"Model fitting failed: {e}")
        return None

def process_package(package_row: pd.Series, args: argparse.Namespace) -> Tuple[Optional[AnalysisResult], Optional[str]]:
    package_name = package_row['package_name']
    session = requests_retry_session()
    
    deprecation_msg, deprecation_ts = get_deprecation_info(session, package_name, args.cache)
    if not deprecation_ts:
        return None, deprecation_msg if deprecation_msg else None
    
    try:
        deprecation_date = isoparse(deprecation_ts)
        start_date_obj = (deprecation_date + relativedelta(months=1)).replace(day=1)
        if start_date_obj.strftime('%Y-%m-%d') >= args.end_date:
            return None, None 
        start_date_str = start_date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return None, "Failed to parse deprecation timestamp"

    monthly_downloads, error = get_monthly_downloads(session, package_name, start_date_str, args.end_date, args.cache)
    if error:
        return None, error

    if monthly_downloads:
        trend_results = analyze_trend(monthly_downloads, args.iqr_factor)
        if trend_results:
            slope, p_value, r_squared = trend_results
            if slope > 0 and p_value < 0.05:
                return AnalysisResult(
                    package_name=package_name,
                    deprecation_date=deprecation_date.strftime('%Y-%m-%d'),
                    deprecation_message=deprecation_msg,
                    slope=slope,
                    p_value=p_value,
                    r_squared=r_squared,
                    npm_url=package_row['npm_url']
                ), None
            else:
                return None, None
        else:
            return None, None
    return None, None

def generate_reports(results: List[AnalysisResult], failed_packages: Dict[str, str], report_md: str, report_csv: str):
    results_sorted = sorted(results, key=lambda x: x.slope, reverse=True)
    
    md_content = f"# NPM Deprecated Package Download Trend Analysis Report (Rebuttal Final Version)\n\n"
    md_content += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if results:
        md_content += "The following packages show a **statistically significant upward trend** (p < 0.05) in monthly downloads after deprecation.\n\n"
        md_content += "| Rank | Package | Deprecation Date | Reason | Log Growth Rate (Slope) | Adj. P-value | R² | Link |\n"
        md_content += "|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|\n"
        for i, res in enumerate(results_sorted):
            package_link = f"[{res.package_name}]({res.npm_url})"
            slope_str = f"`{res.slope:,.4f}`"; pval_str = f"`{res.p_value:.4f}`"; r2_str = f"`{res.r_squared:.3f}`"
            md_content += f"| {i+1} | {package_link} | {res.deprecation_date} | {res.deprecation_message} | {slope_str} | {pval_str} | {r2_str} | [NPM]({res.npm_url}) |\n"
    else:
        md_content += "No packages found with a statistically significant upward trend.\n"

    if failed_packages:
        md_content += "\n---\n\n## Failed Packages\n\n"
        md_content += "| Package | Reason |\n"
        md_content += "|:---|:---|\n"
        for pkg, reason in sorted(failed_packages.items()):
            md_content += f"| `{pkg}` | {reason} |\n"
        
        try:
            with open(FAILED_LOG_FILE, 'w', encoding='utf-8') as f:
                for pkg in sorted(failed_packages.keys()):
                    f.write(f"{pkg}\n")
            logging.info(f"Failed list written to: {FAILED_LOG_FILE}")
        except IOError as e:
            logging.error(f"Error writing failure log: {e}")

    try:
        with open(report_md, 'w', encoding='utf-8') as f: f.write(md_content)
        logging.info(f"Markdown report generated: {report_md}")
    except IOError as e: logging.error(f"Error writing Markdown report: {e}")

    if results:
        try:
            pd.DataFrame([res.__dict__ for res in results_sorted]).to_csv(report_csv, index=False, encoding='utf-8-sig')
            logging.info(f"CSV report generated: {report_csv}")
        except IOError as e: logging.error(f"Error writing CSV report: {e}")

def main(args):
    logging.info(f"Starting analysis with Log-transformation and HAC errors...")
    
    if args.retry_failed:
        logging.info(f"--- Retry Mode Active ---")
        if not os.path.exists(FAILED_LOG_FILE):
            logging.error(f"Error: Log file '{FAILED_LOG_FILE}' not found.")
            return
        try:
            with open(FAILED_LOG_FILE, 'r', encoding='utf-8') as f:
                failed_package_list = [line.strip() for line in f if line.strip()]
            logging.info(f"Loading {len(failed_package_list)} packages from log.")
        except IOError as e:
            logging.error(f"Error reading failure log: {e}")
            return
    else:
        if os.path.exists(FAILED_LOG_FILE):
            os.remove(FAILED_LOG_FILE)
            logging.info(f"Cleared old log: {FAILED_LOG_FILE}")
    
    try:
        df = pd.read_csv(args.input)
        required_columns = {'package_name', 'npm_url', 'last_month_downloads'}
        if not required_columns.issubset(df.columns):
            logging.error(f"Error: CSV missing columns: {', '.join(required_columns)}")
            return
    except FileNotFoundError:
        logging.error(f"Error: Input file '{args.input}' not found.")
        return

    if args.retry_failed:
        df_filtered = df[df['package_name'].isin(failed_package_list)].reset_index(drop=True)
    else:
        df_filtered = df[df['last_month_downloads'] > args.threshold].reset_index(drop=True)
    
    if args.limit > 0:
        df_filtered = df_filtered.head(args.limit)
    
    if df_filtered.empty:
        logging.warning("No packages to process.")
        return
        
    logging.info(f"Analyzing {len(df_filtered.index)} packages.")
    
    successful_results: List[AnalysisResult] = []
    failed_packages: Dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_package = {executor.submit(process_package, row, args): row for _, row in df_filtered.iterrows()}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_package), total=len(df_filtered), desc="Processing"):
            package_row = future_to_package[future]
            package_name = package_row['package_name']
            try:
                result, error_msg = future.result()
                if result:
                    successful_results.append(result)
                elif error_msg:
                    failed_packages[package_name] = error_msg
            except Exception as exc:
                failed_packages[package_name] = f"Unexpected Error: {exc}"

    logging.info("=" * 50)
    generate_reports(successful_results, failed_packages, args.report_md, args.report_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze NPM deprecated package download trends.")
    parser.add_argument('-i', '--input', type=str, default='npm_package_report.csv', help='Input CSV')
    parser.add_argument('-t', '--threshold', type=int, default=10000, help='Download threshold')
    parser.add_argument('-e', '--end-date', type=str, default='2025-06-30', help='Observation end date')
    parser.add_argument('-w', '--workers', type=int, default=10, help='Thread count')
    parser.add_argument('-l', '--limit', type=int, default=0, help='Limit processing count')
    parser.add_argument('--report-md', type=str, default='npm_deprecated_packages_analysis_report.md', help='Output Markdown')
    parser.add_argument('--report-csv', type=str, default='GDNPs.csv', help='Output CSV')
    parser.add_argument('--no-cache', action='store_false', dest='cache', help='Disable cache')
    parser.add_argument('--iqr-factor', type=float, default=1.5, help='IQR multiplier for outliers')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed packages only')
    
    cli_args = parser.parse_args()
    main(cli_args)