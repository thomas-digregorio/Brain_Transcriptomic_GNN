import requests
import xml.etree.ElementTree as ET

BASE_URL = "https://sea-ad-single-cell-profiling.s3.amazonaws.com/"
NAMESPACE = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}

def check_prefix(prefix):
    url = f"{BASE_URL}?list-type=2&prefix={prefix}"
    print(f"Checking {prefix}...")
    try:
        r = requests.get(url)
        root = ET.fromstring(r.content)
        files = []
        for c in root.findall('s3:Contents', NAMESPACE):
            key = c.find('s3:Key', NAMESPACE).text
            size = int(c.find('s3:Size', NAMESPACE).text)
            if key.endswith('.h5ad'):
                files.append((key, size / (1024**3))) # GB
        return files
    except Exception as e:
        print(f"Error: {e}")
        return []

prefixes = ['DREAM/', 'MTG/', 'DLPFC/']
found = False
for p in prefixes:
    files = check_prefix(p)
    for f, s in files:
        print(f"Found: {f} ({s:.2f} GB)")
        found = True

if not found:
    print("No .h5ad files found in checked prefixes.")
