
import requests
import re
import sys

URL = 'https://sea-ad-single-cell-profiling.s3.amazonaws.com/'

print(f"Streaming from {URL}...")
try:
    with requests.get(URL, stream=True) as r:
        content = ''
        found = 0
        for chunk in r.iter_content(chunk_size=8192):
            try:
                chunk_dec = chunk.decode('utf-8', errors='ignore')
                content += chunk_dec
                # Check for keys
                matches = re.findall(r'<Key>(.*?\.h5ad)</Key>', content)
                for m in matches:
                    print(m)
                    found += 1
                
                if found > 5:
                    print("Found enough files. Stopping.")
                    break
                
                # Keep buffer small-ish to avoid memory issues, but need to handle split tags
                if len(content) > 100000:
                    content = content[-1000:] 
            except Exception as e:
                pass
except Exception as e:
    print(f"Error: {e}")
