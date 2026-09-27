from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about rohit sharma"

# text = 'my name is priyanshu yadav'

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# print(cosine_similarity([query_embedding],doc_embedding)) # [query_embedding] both arguments must be in 2d

scores = cosine_similarity([query_embedding],doc_embedding)[0]


index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print('Similarity score is:',score)
# vectors = embedding.embed_query(text)
# print(str(vectors))

# output 
'''tell me about rohit sharma
Rohit Sharma is known for his elegant batting and record-breaking double centuries.
Similarity score is: 0.7273420698471195'''