import requests

resp = requests.post("<google-cloud-run-url>", files={'file': open('zero.jpg', 'rb')})

print(resp.json())
