import asyncio
import aiohttp
import csv
import os
import json
import sys
from tqdm import tqdm
import time

ALL_PACKAGES_URLS = [
    "https://gitee.com/mirrors/all-the-package-names/raw/master/names.json",
    "https://cdn.jsdelivr.net/npm/all-the-package-names/names.json",
    "https://unpkg.com/all-the-package-names/names.json"
]
LOCAL_LIST_FILENAME = "names.json"
DEPRECATED_LIST_FILENAME = 'deprecated_npm_packages.csv'
PROCESSED_LOG_FILENAME = 'processed.log'
FAILED_LOG_FILENAME = 'failed_packages.csv' 

CONCURRENCY_LIMIT = 10
REQUEST_TIMEOUT_SECONDS = 20
IO_BATCH_SIZE = 1000

REGISTRY_API_URL = "https://registry.npmjs.org/{package_name}"
MAX_RETRIES = 3
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)


def read_lines_from_file(filename):
    if not os.path.exists(filename):
        return set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"Warning: Error reading '{filename}': {e}. Treating as empty.")
        return set()

def read_packages_from_csv(filename):
    if not os.path.exists(filename):
        return set()
    packages = set()
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                return set()
            for row in reader:
                if row:
                    packages.add(row[0])
    except Exception as e:
        print(f"Warning: Error reading CSV '{filename}': {e}. Treating as empty.")
        return set()
    return packages

async def is_package_deprecated_async(session: aiohttp.ClientSession, package_name: str, semaphore: asyncio.Semaphore) -> tuple:
    async with semaphore:
        await asyncio.sleep(0.1)
        for attempt in range(MAX_RETRIES):
            try:
                safe_package_name = aiohttp.helpers.quote(package_name)
                url = REGISTRY_API_URL.format(package_name=safe_package_name)
                async with session.get(url, timeout=REQUEST_TIMEOUT) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", "30"))
                        print(f"\n[!] 429 Rate Limit. Waiting {retry_after}s before retry...")
                        await asyncio.sleep(retry_after)
                        raise aiohttp.ClientConnectionError(f"Rate limited (429)")

                    if response.status == 404:
                        return (package_name, False, True, "Not Found")
                    
                    response.raise_for_status()
                    data = await response.json(content_type=None)
                    
                    is_deprecated = False
                    if isinstance(data, dict):
                        if 'deprecated' in data:
                            is_deprecated = True
                        else:
                            latest_version_tag = data.get('dist-tags', {}).get('latest')
                            if latest_version_tag:
                                latest_version_data = data.get('versions', {}).get(latest_version_tag, {})
                                if 'deprecated' in latest_version_data:
                                    is_deprecated = True
                    return (package_name, is_deprecated, True, "Success")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error_reason = type(e).__name__
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** (attempt + 1)
                    await asyncio.sleep(wait_time)
                else:
                    return (package_name, False, False, error_reason)
                    
    return (package_name, False, False, "UnknownLogicError")

def write_csv_batch(filename, batch, header):
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(batch)

async def main():
    print("--- Step 1: Loading Package List ---")
    if not os.path.exists(LOCAL_LIST_FILENAME):
        print(f"Error: Local file '{LOCAL_LIST_FILENAME}' not found. Please download it first.")
        return
    with open(LOCAL_LIST_FILENAME, 'r', encoding='utf-8') as f:
        all_packages = json.load(f)
    
    print("--- Step 2: Loading Progress for Incremental Scan ---")
    processed_packages = read_lines_from_file(PROCESSED_LOG_FILENAME)
    failed_packages = read_packages_from_csv(FAILED_LOG_FILENAME)
    deprecated_packages_in_csv = read_packages_from_csv(DEPRECATED_LIST_FILENAME)

    already_handled = processed_packages.union(failed_packages).union(deprecated_packages_in_csv)
    packages_to_process = [p for p in all_packages if p not in already_handled]
    
    found_count = len(deprecated_packages_in_csv)
    total_to_scan = len(packages_to_process)
    
    print(f"History loaded.")
    print(f"Total packages in list: {len(all_packages):,}")
    print(f"Already handled: {len(already_handled):,}")
    print(f"New packages to scan: {total_to_scan:,}")
    print(f"Deprecated packages found so far: {found_count:,}")

    if not packages_to_process:
        print("\nAll packages scanned. Exiting.")
        return

    print(f"\n--- Step 3: Starting Scan (Concurrency: {CONCURRENCY_LIMIT}, Timeout: {REQUEST_TIMEOUT_SECONDS}s) ---")
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    processed_batch = []
    failed_batch = []
    deprecated_batch = []
    fail_count = 0
    start_time = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [is_package_deprecated_async(session, name, semaphore) for name in packages_to_process]
            pbar = tqdm(asyncio.as_completed(tasks), total=total_to_scan, desc="Scanning")
            for future in pbar:
                package_name, is_deprecated, status_ok, reason = await future

                if status_ok:
                    processed_batch.append(package_name)
                    if is_deprecated:
                        deprecated_batch.append([package_name])
                        found_count += 1
                else:
                    failed_batch.append([package_name, reason])
                    fail_count += 1

                pbar.set_postfix_str(f"Deprecated: {found_count}, Failed: {fail_count}")

                if len(processed_batch) >= IO_BATCH_SIZE:
                    with open(PROCESSED_LOG_FILENAME, 'a', encoding='utf-8') as f: 
                        f.writelines(f"{p}\n" for p in processed_batch)
                    processed_batch.clear()

                if len(failed_batch) >= IO_BATCH_SIZE:
                    write_csv_batch(FAILED_LOG_FILENAME, failed_batch, ['Name', 'Reason'])
                    failed_batch.clear()
                
                if len(deprecated_batch) >= IO_BATCH_SIZE:
                    write_csv_batch(DEPRECATED_LIST_FILENAME, deprecated_batch, ['Name'])
                    deprecated_batch.clear()

    finally:
        print("\nFinalizing and saving data...")
        if processed_batch:
            with open(PROCESSED_LOG_FILENAME, 'a', encoding='utf-8') as f: 
                f.writelines(f"{p}\n" for p in processed_batch)
        if failed_batch:
            write_csv_batch(FAILED_LOG_FILENAME, failed_batch, ['Name', 'Reason'])
        if deprecated_batch:
            write_csv_batch(DEPRECATED_LIST_FILENAME, deprecated_batch, ['Name'])
        
        end_time = time.time()
        duration = end_time - start_time
        scanned_count = pbar.n 
        speed = scanned_count / duration if duration > 0 else 0

        print(f"\n--- Scan Completed ---")
        print(f"Time: {duration:.2f}s, Processed: {scanned_count:,}, Avg Speed: {speed:.2f} pkg/s")
        print(f"Total Deprecated Found: {found_count:,}")
        print(f"Batch Failures: {fail_count:,} (See '{FAILED_LOG_FILENAME}')")

if __name__ == "__main__":
    if sys.platform.startswith('win') and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())