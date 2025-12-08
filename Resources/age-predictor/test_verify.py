import requests
import json

# Test the verify-face endpoint
url = "http://localhost:5005/verify-face"

# Use a test image file
files = {'file': open('test_face.jpg', 'rb')}  # Replace with actual image path

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", e)