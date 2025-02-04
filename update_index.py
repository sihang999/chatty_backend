import os
import json
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters.markdown import MarkdownTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
import sentence_model

data_folder_path = "markdown_files"
status_file = "file_status.json"
vector_store_path = "faiss_index"


def ensure_files_and_folders_exist(data_folder_path= "markdown_files", status_file="file_status.json"):
    """
    Ensure the specified folder and files exist. If not, create them.

    Parameters:
        data_folder_path (str): Path to the data folder.
        status_file (str): Path to the status file.
    """
    # Ensure the data folder exists
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
        print(f"Folder '{data_folder_path}' created.")
    else:
        print(f"Folder '{data_folder_path}' already exists.")


    # Ensure the status file exists
    if not os.path.exists(status_file):
        with open(status_file, 'w') as file:
            json.dump({}, file)  # Initialize with an empty JSON object
        print(f"File '{status_file}' created.")
    else:
        print(f"File '{status_file}' already exists.")


#  Just get the existed vector_store
def get_verctor_store(vector_store_path = "faiss_index"):
    # 创建一个嵌入模型实例，用于将文本转换为向量（数值表示）。
    # embeddings = HuggingFaceEmbeddings(model_name="./multi-qa-mpnet-base-dot-v1")
    index = faiss.IndexFlatL2(len(sentence_model.EMBEDDINGS_MODELv1.embed_query("hello world")))
    # 初始化vector store
    vector_store = FAISS(
            embedding_function=sentence_model.EMBEDDINGS_MODELv1,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
    #  加载已有的向量存储
    if os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(
            "faiss_index", sentence_model.EMBEDDINGS_MODELv1, allow_dangerous_deserialization=True
        )
    return vector_store



# update the new files index and delete the no existed files index

def manage_files(data_folder_path = "markdown_files", status_file = "file_status.json", vector_store_path = "faiss_index"):

    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    vector_store = get_verctor_store()

    # read the file last change time and uuid 
    if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
        with open(status_file, "r") as f:
            file_status = json.load(f)
    else:
        file_status = {}

    # copy the file status for new status
    new_file_status = file_status.copy()
    # default there is no change in markdown_files
    change=False

    # get the current files names list
    current_files = set(os.listdir(data_folder_path))

    # check if there is any changed files or new files, only handle md files
    for file_name in os.listdir(data_folder_path):
        if file_name.endswith(".md"):
            file_path = os.path.join(data_folder_path, file_name)
            file_mtime = os.path.getmtime(file_path)  
            file_id = file_name  

            # 根据创建时间检查文件是否更新，根据文件名称检查文件是否被新创建
            if file_id not in file_status or file_status[file_id]["mtime"] != file_mtime:
                print(f"Processing changed or new file: {file_name}")

                # loade the new file or changed file and split them 
                loader = UnstructuredMarkdownLoader(file_path, mode="single", strategy="fast")
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)

                # delete the old vector of new file
                if file_id in file_status:
                    print(f"Deleting old vectors for {file_name}")
                    old_uuids = file_status[file_id]["uuids"]
                    vector_store.delete(ids=old_uuids)

                # generate the new uuid for files
                new_uuids = [str(uuid4()) for _ in range(len(split_docs))]
                vector_store.add_documents(documents=split_docs, ids=new_uuids)
                print(f"Added {len(split_docs)} documents for {file_name}")

                # add the file status to the list
                new_file_status[file_id] = {
                    "mtime": file_mtime,
                    "uuids": new_uuids
                }
                change=True

            else:
                print(f"No changes detected for {file_name}")
    # check there is any old file deleted
    for file_id in list(file_status.keys()):

        if file_id not in current_files:  
            print(f"File deleted: {file_id}")
            
            # delete the according vectors
            old_uuids = file_status[file_id]["uuids"]
            vector_store.delete(ids=old_uuids)
            print(f"Deleted vectors for {file_id} in Faiss index")

            # delete the according file status
            del new_file_status[file_id]
            print(f"Deleted record for {file_id} in file_status")
            change = True
    if change:
        # store the new vector_store
        vector_store.save_local(vector_store_path)

        # store the new status_files
        with open(status_file, "w") as f:
            json.dump(new_file_status, f)
        print("Update complete!")




# # # test above function
# if __name__ == "__main__":
#     ensure_files_and_folders_exist()
#     print("Starting manage_files...")
#     manage_files()

