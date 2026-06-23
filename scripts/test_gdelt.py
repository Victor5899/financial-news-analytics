# scripts/test_gdelt.py

import requests

url = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query=Apple"
    "&mode=ArtList"
    "&maxrecords=10"
    "&format=json"
)

response = requests.get(url)

print(response.status_code)
print(response.text[:500])