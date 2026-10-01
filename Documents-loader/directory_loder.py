from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = 'ml book',
    glob='**pdf', # load pdf file form ml book , for text "**/*.txt", *.csv
    loader_cls = PyPDFLoader
)
# ml_book = loader()
ml_book = loader.lazy_load()

# print(len(ml_book)) # Total pdf page --> 1152
"""print(ml_book[0].page_content)
print(ml_book[0].metadata)"""

for doctuments in ml_book:
    print(doctuments.metadata)

""" load() reads all documents into memory at once as a list—best for small files.
lazy_load() streams documents one at a time on demand using a generator—best for large datasets to save memory. """