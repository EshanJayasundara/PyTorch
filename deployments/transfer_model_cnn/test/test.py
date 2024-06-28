import requests

resp = requests.post("https://getprediction-pkb4z24wzq-de.a.run.app", files={'file': open('zero.jpg', 'rb')})

print(resp.json())