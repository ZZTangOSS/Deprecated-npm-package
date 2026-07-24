import os
import time
import threading
import logging
from itertools import cycle
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

MAX_WORKERS = 10
INPUT_CSV = "new_age_download_with_ADNP_cleaned.csv"
DATE_CSV = "GDNPs.csv"

BASE_DIR = Path(__file__).resolve().parent
DONE_MARKERS_DIR = BASE_DIR / "done_markers"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_writer_lock = threading.Lock()

class TokenManager:
    def __init__(self):
        dotenv_path = BASE_DIR / ".env"
        if not dotenv_path.exists(): raise FileNotFoundError(f"error:no .env file found at {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
        self._tokens = [token for name, token in os.environ.items() if name.startswith("GITHUB_TOKEN")]
        if not self._tokens: raise ValueError("error: no 'GITHUB_TOKEN' variable found in .env file")
        logging.info(f"success: loaded {len(self._tokens)} GitHub API tokens.")
        self._lock = threading.Lock()
        self._token_cycler = cycle(self._tokens)
        self._ratelimited_tokens = {}

    def _release_expired_tokens(self):
        current_time = time.time()
        released = [token for token, reset in self._ratelimited_tokens.items() if current_time >= reset]
        for token in released:
            del self._ratelimited_tokens[token]
            self._tokens.append(token)
            logging.info(f"success: token ...{token[-4:]} has been released from rate limit.")
        if released: self._token_cycler = cycle(self._tokens)

    def get_token(self) -> str:
        with self._lock:
            self._release_expired_tokens()
            while not self._tokens:
                if not self._ratelimited_tokens:
                    raise RuntimeError("error: all tokens are exhausted and no tokens are available for release.")
                wait_time = min(self._ratelimited_tokens.values()) - time.time() + 1
                logging.warning(f"warning: all tokens are rate-limited. Pausing for {wait_time:.1f} seconds...")
                self._lock.release()
                time.sleep(max(0, wait_time))
                self._lock.acquire()
                self._release_expired_tokens()
            return next(self._token_cycler)

    def report_ratelimit(self, token: str, reset_timestamp: int):
        with self._lock:
            if token in self._tokens:
                self._tokens.remove(token)
                self._ratelimited_tokens[token] = reset_timestamp
                logging.warning(f"warning: token ...{token[-4:]} has reached its rate limit. It will be reset at {time.ctime(reset_timestamp)}.")
                if self._tokens: self._token_cycler = cycle(self._tokens)

def make_api_request(url: str, token_manager: TokenManager, params: dict = None) -> requests.Response:
    max_retries, base_wait_time = 5, 5
    for attempt in range(max_retries):
        token = token_manager.get_token()
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
            if response.status_code in [403, 429]:
                if response.headers.get('X-RateLimit-Remaining') == '0':
                    reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                    token_manager.report_ratelimit(token, reset_time)
                    continue
                elif 'retry-after' in response.headers:
                    retry_after = int(response.headers.get('retry-after'))
                    logging.warning(f"warning: triggered secondary rate limit. Current token needs cooling for {retry_after} seconds.")
                    token_manager.report_ratelimit(token, int(time.time()) + retry_after)
                    continue
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"error: request failed: {e}.")
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (2 ** attempt)
                logging.info(f"info: will retry in {wait_time} seconds...")
                time.sleep(wait_time)
            else: raise
    raise RuntimeError("error: request loop exited")

def is_github_bot(user_dict: dict) -> bool:
    if not user_dict: return False
    user_type = user_dict.get('type', '')
    login_name = user_dict.get('login', '').lower()
    if user_type == 'Bot':
        return True
    known_bots = ['dependabot', 'greenkeeper', 'renovate', 'snyk', 'github-actions']
    if login_name.endswith('[bot]') or login_name.endswith('-bot') or any(bot in login_name for bot in known_bots):
        return True
    return False

def get_all_issue_numbers(owner: str, repo: str, token_manager: TokenManager) -> list[int]:
    all_issue_numbers, url, page_num = [], f"https://api.github.com/repos/{owner}/{repo}/issues", 1
    params = {"state": "all", "per_page": 100}
    logging.info(f"starting to fetch all issue numbers for {owner}/{repo}...")
    while url:
        logging.info(f"starting to fetch page {page_num}...")
        try:
            response = make_api_request(url, token_manager, params=(params if page_num == 1 else None))
            page_data = response.json()
            if not page_data: break
            all_issue_numbers.extend(item['number'] for item in page_data if "pull_request" not in item)
            url = response.links.get('next', {}).get('url')
            page_num += 1
        except requests.exceptions.RequestException as e:
            logging.error(f"error: failed to fetch issue numbers: {e}")
            break
    logging.info(f"successfully fetched {len(all_issue_numbers)} issue numbers for {owner}/{repo}.")
    return all_issue_numbers
    
def fetch_and_parse_comments(owner: str, repo: str, issue_number: int, token_manager: TokenManager, target_start: pd.Timestamp, target_end: pd.Timestamp) -> list[dict]:
    comments_list = []
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
    params = {"per_page": 100}
    while url:
        try:
            response = make_api_request(url, token_manager, params=params if not comments_list else None)
            comments_data = response.json()
            if not comments_data: break
            for comment in comments_data:
                created_at = comment.get('created_at', '')
                if not created_at:
                    continue
                
                dt = pd.to_datetime(created_at, utc=True)
                if not (target_start <= dt < target_end):
                    continue
                
                user_info = comment.get('user', {})
                if is_github_bot(user_info):
                    continue 
                comments_list.append({
                    'comment_id': comment.get('id'), 'issue_number': issue_number,
                    'author': user_info.get('login'), 'created_at': created_at,
                    'updated_at': comment.get('updated_at'),
                    'body': (comment.get('body', '') or "").replace('\r', ' ').replace('\n', ' ').strip(),
                })
            url = response.links.get('next', {}).get('url')
        except requests.exceptions.RequestException as e:
            logging.error(f"error: failed to fetch comments for Issue #{issue_number}: {e}")
            break
    return comments_list

def process_single_issue(issue_number: int, owner: str, repo: str, token_manager: TokenManager, target_start: pd.Timestamp, target_end: pd.Timestamp) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    try:
        response = make_api_request(url, token_manager)
        data = response.json()
        user_info = data.get('user', {})
        if is_github_bot(user_info):
            return {'status': 'skipped_bot', 'number': issue_number, 'reason': f"Bot issue by {user_info.get('login')}"}
        
        created_at = data.get('created_at', '')
        is_valid_date = False
        if created_at:
            dt = pd.to_datetime(created_at, utc=True)
            is_valid_date = bool(target_start <= dt < target_end)
        
        comments_data = []
        if data.get('comments', 0) > 0:
            comments_data = fetch_and_parse_comments(owner, repo, issue_number, token_manager, target_start, target_end)

        if not is_valid_date and not comments_data:
            return {'status': 'skipped_date', 'number': issue_number, 'reason': 'Out of date range'}
            
        issue_body = (data.get('body', '') or "").replace('\r', ' ').replace('\n', ' ').strip()[:30000] if is_valid_date else ""
        issue_title = data.get('title', '') if is_valid_date else ""
        
        issue_data = {
            'number': data.get('number'), 'id': data.get('id'), 'title': issue_title,
            'author': user_info.get('login'), 'state': data.get('state'),
            'created_at': created_at, 'updated_at': data.get('updated_at'),
            'closed_at': data.get('closed_at'), 'comments_count': data.get('comments'),
            'labels': ", ".join([label['name'] for label in data.get('labels', [])]),
            'body': issue_body,
        }
        
        return {'status': 'success', 'issue_data': issue_data, 'comments_data': comments_data}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [404, 410]: return {'status': 'not_found', 'number': issue_number}
        return {'status': 'failed', 'number': issue_number, 'reason': str(e)}
    except Exception as e:
        return {'status': 'failed', 'number': issue_number, 'reason': str(e)}

def sanitize_filename(name: str) -> str:
    return name.replace('/', '__')

def parse_github_url(url: str) -> tuple[str, str] | None:
    try:
        if not isinstance(url, str) or not url.startswith("http"): return None
        parsed_path = urlparse(url).path.strip('/').split('/')
        if len(parsed_path) >= 2: 
            repo_name = parsed_path[1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            return parsed_path[0], repo_name
        return None
    except Exception as e:
        logging.warning(f"error: failed to parse URL '{url}': {e}")
        return None

def standardize_github_url(url: str) -> str:
    res = parse_github_url(url)
    if res:
        return f"{res[0].lower()}/{res[1].lower()}"
    return str(url).strip().lower()
        
def append_df_to_csv(df: pd.DataFrame, filepath: Path):
    with file_writer_lock:
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            header = not filepath.exists() or os.path.getsize(filepath) == 0
            df.to_csv(filepath, mode='a', header=header, index=False, encoding='utf-8-sig')
        except IOError as e: 
            logging.error(f"error: failed to write to file {filepath}: {e}")
            
def load_processed_ids(filepath: Path, id_column: str = 'number') -> set[int]:
    if not filepath.exists(): return set()
    try:
        df = pd.read_csv(filepath); return set(df[id_column].unique())
    except (pd.errors.EmptyDataError, KeyError): return set()
    except Exception as e: logging.error(f"error: failed to load processed IDs from {filepath}: {e}"); return set()

def load_not_found_ids(filepath: Path) -> set[int]:
    if not filepath.exists(): return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return {int(line.strip()) for line in f if line.strip()}
    except (IOError, ValueError) as e: logging.error(f"error: failed to load {filepath}: {e}"); return set()
    
def log_failure(issue_number: int, reason: str, filepath: Path):
    df = pd.DataFrame([{'number': issue_number, 'reason': reason, 'timestamp': time.time()}])
    append_df_to_csv(df, filepath)

def log_not_found(issue_number: int, filepath: Path):
    with file_writer_lock:
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'a', encoding='utf-8') as f: f.write(f"{issue_number}\n")
        except IOError as e: logging.error(f"error: failed to write to Not Found log file {filepath}: {e}")
        
def process_repository(owner: str, repo: str, safe_package_name: str, token_manager: TokenManager, target_start: pd.Timestamp, target_end: pd.Timestamp):
    output_issues_csv = BASE_DIR / f"{safe_package_name}_issues.csv"
    output_comments_csv = BASE_DIR / f"{safe_package_name}_comments.csv"
    failed_log_csv = BASE_DIR / f"{safe_package_name}_failed_log.csv"
    not_found_log = BASE_DIR / f"{safe_package_name}_not_found_log.txt"
    done_marker_path = DONE_MARKERS_DIR / f"{safe_package_name}.done"
    
    processed_ids = load_processed_ids(output_issues_csv)
    not_found_ids = load_not_found_ids(not_found_log)
    failed_to_retry_ids = load_processed_ids(failed_log_csv)
    
    all_issue_numbers = get_all_issue_numbers(owner, repo, token_manager)
    
    if not all_issue_numbers:
        done_marker_path.touch()
        return

    target_ids = (set(all_issue_numbers) - (processed_ids | not_found_ids)) | failed_to_retry_ids
    
    if not target_ids:
        done_marker_path.touch()
        return

    issues_to_process = sorted(list(target_ids))
    counts = {'success': 0, 'failed': 0, 'not_found': 0, 'skipped_bot': 0, 'skipped_date': 0}
    
    worker_func = partial(process_single_issue, owner=owner, repo=repo, token_manager=token_manager, target_start=target_start, target_end=target_end)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker_func, num): num for num in issues_to_process}
        pbar = tqdm(as_completed(futures), total=len(issues_to_process), desc=f"处理 {owner}/{repo}")
        for future in pbar:
            result = future.result()
            status = result['status']
            counts[status] += 1
            if status == 'success':
                append_df_to_csv(pd.DataFrame([result['issue_data']]), output_issues_csv)
                if result['comments_data']:
                    append_df_to_csv(pd.DataFrame(result['comments_data']), output_comments_csv)
            elif status == 'failed':
                log_failure(result['number'], result['reason'], failed_log_csv)
            elif status == 'not_found':
                log_not_found(result['number'], not_found_log)
            pbar.set_postfix(success=counts['success'], failed=counts['failed'], bots=counts['skipped_bot'], skipped=counts['skipped_date'])

    remaining_failures = load_processed_ids(failed_log_csv)
    
    if failed_log_csv.exists():
        successful_retries = failed_to_retry_ids - remaining_failures
        if successful_retries or not remaining_failures:
            try:
                df_failed = pd.read_csv(failed_log_csv)
                df_failed = df_failed[df_failed['number'].isin(remaining_failures)]
                if df_failed.empty:
                    failed_log_csv.unlink()
                else:
                    df_failed.to_csv(failed_log_csv, index=False, encoding='utf-8-sig')
            except Exception as e:
                logging.error(f"error: failed to clean failed log file {failed_log_csv}: {e}")

    if not remaining_failures:
        done_marker_path.touch()

if __name__ == "__main__":
    try:
        DONE_MARKERS_DIR.mkdir(exist_ok=True)
        token_manager = TokenManager()
        
        input_csv_path = BASE_DIR / INPUT_CSV
        date_csv_path = BASE_DIR / DATE_CSV
        
        if not input_csv_path.exists():
            raise FileNotFoundError(f"error: input file {input_csv_path} not found.")
        if not date_csv_path.exists():
            raise FileNotFoundError(f"error: deprecation date dictionary file {date_csv_path} not found.")
            
        repo_df = pd.read_csv(input_csv_path)
        date_df = pd.read_csv(date_csv_path)

        date_mapping = dict(zip(date_df['package_name'], date_df['deprecation_date']))

        repo_df['standardized_repo'] = repo_df['repository'].apply(standardize_github_url)
        repo_df.drop_duplicates(subset=['standardized_repo'], keep='first', inplace=True)
        
        for index, row in repo_df.iterrows():
            package_name = row.get('package_name')
            github_url = row.get('repository')

            deprecation_date_str = str(date_mapping.get(package_name, '')).strip()

            if not package_name or pd.isna(github_url) or not deprecation_date_str or deprecation_date_str == 'nan':
                continue
                
            try:
                target_start = pd.to_datetime(deprecation_date_str, utc=True)
                target_end = pd.to_datetime("2025-07-14", utc=True)
            except Exception as e:
                logging.warning(f"error: failed to parse deprecation date for package {package_name}: {e}")
                continue

            safe_package_name = sanitize_filename(package_name)
            done_marker_path = DONE_MARKERS_DIR / f"{safe_package_name}.done"
            if done_marker_path.exists():
                continue
            
            repo_info = parse_github_url(github_url)
            if not repo_info:
                continue
            
            owner, repo = repo_info
            process_repository(owner, repo, safe_package_name, token_manager, target_start, target_end)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logging.error(f"error: failed to start or execute the program: {e}")
    except Exception as e:
        logging.error(f"error: an unknown error occurred: {e}")