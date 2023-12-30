# from dotenv import load_dotenv
# load_dotenv()

from openai import OpenAI

api_key = 'sess-p0K8EftLSuJgiN4MIYqzhw3IvesR6lzuyMd1zaS5'

client = OpenAI(api_key=api_key)

response = client.images.generate(
    model="dall-e-2",
    prompt="kim jung en",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
print(image_url)