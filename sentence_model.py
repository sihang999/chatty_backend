from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDINGS_MODELv1 = None

def get_model():
    global EMBEDDINGS_MODELv1
    # EMBEDDINGS_MODELv1 = HuggingFaceEmbeddings(model_name="./multi-qa-mpnet-base-dot-v1")
    EMBEDDINGS_MODELv1 = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

