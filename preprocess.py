from loguru import logger
from tqdm import tqdm
from utils.vectorizer import EmbeddingProcessor
from inference import ChatGPT


def upsert_all_by_txt(txt: str = None):
    """
    将知识库存入db中
    """
    emb_proc = EmbeddingProcessor(embedding_db="./assets/test-vector-db.jsonl")

    # for embedding
    chat_gpt = ChatGPT()

    # build vector db
    vectors = []
    with open(txt, encoding="utf-8", mode="r") as reader:
        pbar = tqdm(reader.readlines(), desc=f"Reading from source and start embedding")
        for line in pbar:
            text = line.strip()

            # build db data format
            # data = [{'embedding': [1.0, 2.0, 3.0], 'text': "你好吗", 'metadata': {'key': 'value'}},
            #         {'embedding': [4.0, 5.0, 6.0], 'text': "我很好", 'metadata': {'key': 'value'}}]
            try:
                vector = {'embedding': chat_gpt.start_embedding(text), 'text': text, 'metadata': {'key': 'value'}}
                vectors.append(vector)
            except Exception as e:
                logger.exception(e)
            pbar.update(1)

    # upsert all vectors
    emb_proc.upsert(vectors)


if __name__ == "__main__":
    upsert_all_by_txt("./data/latest.txt")
