import os
from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[
      {"role": "system", "content": "You are a helpful assistant." },
      {"role":"user","content":"Write a short poet about the beauty of Roses."}
    ],
  temperature=0.2,
  top_p=0.7,
  max_tokens=512,
  stream=False
)

# Print the streaming response  
# for chunk in completion:
#    if chunk.choices[0].finish_reason is not None:
#        break
#   if chunk.choices[0].delta.get("content"):
#        print(chunk.choices[0].delta["content"], end="", flush=True)

print(completion.choices[0].message)
