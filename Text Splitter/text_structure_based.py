from langchain_text_splitters import RecursiveCharacterTextSplitter

# RecursiveCharacterTextSplitter most important always use it
text = """Machine learning (ML) is a subfield of artificial intelligence (AI) focused on developing algorithms and statistical models that enable computer systems to learn from data and improve performance without explicit programming.  By identifying patterns in training data, ML models generalize to make accurate predictions, classifications, or decisions on new, unseen data"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap = 0
)
chunks = splitter.split_text(text)
print(len(chunks))