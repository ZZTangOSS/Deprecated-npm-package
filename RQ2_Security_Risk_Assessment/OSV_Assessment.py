import requests
import pandas as pd
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from cvss import CVSS3, CVSS2
except ImportError:
    sys.exit(1)

INPUT_CSV = "GDNPs.csv"
OUTPUT_FILE = "GDNPs_assessment.csv" 

OSV_QUERY_API = "https://api.osv.dev/v1/query"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

def create_session():
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=3,
        pool_connections=20,
        pool_maxsize=20
    )
    session.mount('https://', adapter)
    return session

def fetch_osv_data(session, pkg_name):
    try:
        payload = {"package": {"name": pkg_name, "ecosystem": "npm"}}
        resp = session.post(OSV_QUERY_API, json=payload, timeout=15)
        
        if resp.status_code == 200:
            return resp.json().get('vulns', [])
        elif resp.status_code == 404:
            return [] 
        else:
            return []
    except Exception as e:
        return []

def parse_cvss_vector(vector_str, vuln_type="CVSS_V3"):
    score = 0.0
    attack_vector = "Unknown"
    
    try:
        if not vector_str:
            return 0.0, "None"

        if vuln_type == "CVSS_V3":
            c = CVSS3(vector_str)
            score = c.scores()[0]
            if "/AV:N" in vector_str: attack_vector = "Network (Remote)"
            elif "/AV:A" in vector_str: attack_vector = "Adjacent (Local)"
            elif "/AV:L" in vector_str: attack_vector = "Local (Access Required)"
            elif "/AV:P" in vector_str: attack_vector = "Physical"
            
        elif vuln_type == "CVSS_V2":
            c = CVSS2(vector_str)
            score = c.scores()[0]
            if "AV:N" in vector_str: attack_vector = "Network"
            elif "AV:L" in vector_str: attack_vector = "Local"
            elif "AV:A" in vector_str: attack_vector = "Adjacent"

    except Exception:
        pass
    
    return score, attack_vector

def extract_vuln_details(pkg_name, dep_date, vuln_data):
    v_id = vuln_data.get("id", "UNKNOWN")
    summary = vuln_data.get("summary", "No summary provided.")
    
    aliases = vuln_data.get("aliases", [])
    cve_ids = [x for x in aliases if x.startswith("CVE")]
    cve_str = ", ".join(cve_ids) if cve_ids else ""

    published = vuln_data.get("published", "")
    modified = vuln_data.get("modified", "")
    withdrawn = "Yes" if "withdrawn" in vuln_data else "No"

    details = vuln_data.get("details", "")
    details_preview = details[:100].replace("\n", " ") + "..." if details else ""

    references = vuln_data.get("references", [])
    default_url = references[0].get("url", "") if references else f"https://osv.dev/vulnerability/{v_id}"

    db_specific = vuln_data.get("database_specific", {})
    severity_rating = db_specific.get("severity", "").upper()

    cvss_vector = ""
    cvss_score = 0.0
    attack_vector = "Unknown"
    
    severity_list = vuln_data.get("severity", [])
    found_vector = False
    
    for item in severity_list:
        if item.get("type") == "CVSS_V3":
            cvss_vector = item.get("score", "")
            cvss_score, attack_vector = parse_cvss_vector(cvss_vector, "CVSS_V3")
            found_vector = True
            break
            
    if not found_vector:
        for item in severity_list:
            if item.get("type") == "CVSS_V2":
                cvss_vector = item.get("score", "")
                cvss_score, attack_vector = parse_cvss_vector(cvss_vector, "CVSS_V2")
                found_vector = True
                break
    
    if not severity_rating:
        if cvss_score >= 9.0: severity_rating = "CRITICAL"
        elif cvss_score >= 7.0: severity_rating = "HIGH"
        elif cvss_score >= 4.0: severity_rating = "MODERATE"
        elif cvss_score > 0.0: severity_rating = "LOW"
        else: severity_rating = "UNKNOWN"

    return {
        "package_name": pkg_name,
        "cve_ids": cve_str,
        "severity_rating": severity_rating,
        "cvss_score": cvss_score,
        "published_at": published,
        "modified_at": modified,
        "is_withdrawn": withdrawn,
        "summary": summary,
        "details_preview": details_preview,
        "attack_vector": attack_vector,
        "vuln_id": v_id,
        "advisory_url": default_url,
        "deprecation_date": dep_date
    }

def process_package(session, row):
    pkg_name = row['package_name']
    dep_date = row.get('deprecation_date', '')
    
    if not isinstance(pkg_name, str) or not pkg_name.strip():
        return []

    raw_vulns = fetch_osv_data(session, pkg_name)
    if not raw_vulns:
        return []

    processed_rows = []
    seen_ids = set()

    for vuln in raw_vulns:
        v_id = vuln.get("id")
        if v_id in seen_ids:
            continue
        seen_ids.add(v_id)
        
        row_data = extract_vuln_details(pkg_name, dep_date, vuln)
        processed_rows.append(row_data)
        
    return processed_rows

def main():
    try:
        df = pd.read_csv(INPUT_CSV)
        df.columns = [c.strip() for c in df.columns]
        
        if 'package_name' not in df.columns:
            logger.error("Input file missing 'package_name' column")
            return
            
        df_unique = df.drop_duplicates(subset=['package_name'])
        packages = df_unique.to_dict('records')
        logger.info(f"Preparing to analyze {len(packages)} packages...")
        
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    session = create_session()
    all_vulnerabilities = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_package, session, row): row['package_name'] for row in packages}
        
        count = 0
        total = len(packages)
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_vulnerabilities.extend(res)
            
            count += 1
            if count % 10 == 0:
                print(f"\rProgress: {count}/{total} ({(count/total)*100:.1f}%)", end="")
    
    print("\nAnalysis complete.")

    if all_vulnerabilities:
        df_out = pd.DataFrame(all_vulnerabilities)
        
        cols_order = [
            'package_name', 
            'cve_ids',
            'severity_rating', 
            'cvss_score', 
            'summary', 
            'published_at',
            'details_preview',
            'advisory_url',
            'vuln_id',
            'is_withdrawn'
        ]
        
        final_cols = [c for c in cols_order if c in df_out.columns]
        df_out = df_out[final_cols]
        
        if 'published_at' in df_out.columns:
             df_out = df_out.sort_values(by=['cvss_score', 'published_at'], ascending=[False, False])
        else:
             df_out = df_out.sort_values(by=['cvss_score'], ascending=[False])
        
        df_out.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Full audit report saved to: {OUTPUT_FILE}")
        
        print("\n" + "="*100)
        print("High Severity Vulnerability Audit Preview (Top 5)")
        print("="*100)
        pd.set_option('display.max_colwidth', 30) 
        
        preview_cols = ['package_name', 'cve_ids', 'severity_rating', 'cvss_score', 'summary']
        valid_preview_cols = [c for c in preview_cols if c in df_out.columns]
        print(df_out[valid_preview_cols].head(5).to_string(index=False))
        
    else:
        logger.warning("No vulnerabilities found.")

if __name__ == "__main__":
    main()