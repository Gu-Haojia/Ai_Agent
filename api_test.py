from langchain.chat_models import init_chat_model
from qq_group_bot import _load_env_from_files

_load_env_from_files([".env.local", ".env"])

model = init_chat_model("google_genai:gemini-3.5-flash-lite")

print(model.invoke("Hello, world!"))