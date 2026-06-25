import requests

# Test health
response = requests.get("http://localhost:8000/health")
print("Health:", response.json())

# Test analyze with a sample image
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"image": ("test.jpg", f, "image/jpeg")}
    )
    print("Analysis:", response.json())