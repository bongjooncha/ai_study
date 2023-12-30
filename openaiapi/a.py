# 이미지 생성
from openai import OpenAI

# 올바른 API 키를 입력해주세요
api_key = "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"

client = OpenAI(api_key=api_key)


response = client.images.generate(
  model="dall-e-2",
  prompt="cat",
  n=1,
  size="1024x1024"
)

print(response)
