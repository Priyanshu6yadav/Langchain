from langchain_text_splitters import CharacterTextSplitter

text = """LangChain is a powerful framework for developing applications powered by language models.
It enables developers to chain together components like LLMs, prompts, and memory to create advanced conversational AI systems.
Text splitters in LangChain help break large documents into smaller pieces for processing."""

splitter = CharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=10,
    separator=""
)
result = splitter.split_text(text)
print(result)