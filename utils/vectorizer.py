import json
from loguru import logger
from typing import List, Optional, Dict, Any
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingProcessor:
    def __init__(self, embedding_db: Optional[str] = None):
        self.embedding_db = embedding_db

        if self.embedding_db is None:
            raise ValueError("embedding_db can not be None!")

        # load all data one time
        if Path(self.embedding_db).exists():
            self.embeddings_list, self.texts_list = self.load_all_data_to_list()

    def query(self, vector: List[float] = None, top_k: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """
        query the embedding and text from the existed vector db.
        :param vector: the target embedding that will be compared with all embeddings.
        :param top_k: the top k embeddings selected.
        :return:
        """
        source_data = []
        similarity_dict = {}
        with open(self.embedding_db, 'r') as jsonl_reader:
            for index, json_line in enumerate(jsonl_reader):
                source_data.append(json.loads(json_line.strip()))
                embedding_values = json.loads(json_line.strip())["embedding"]
                similarity_dict[index] = cosine_similarity(X=[embedding_values], Y=[vector])[0][0]
            similarity_list = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

            data_num = len(similarity_list)
            if top_k < data_num:
                return [source_data[k] for k, v in similarity_list[:top_k]]
            else:
                return [source_data[k] for k, v in similarity_list]

    def fast_query(self, vector: List[float] = None, top_k: Optional[int] = 2) -> List[Dict[str, Any]]:
        """
        fast query mode, matrix calculate will accelerate the speed of query.
        :param vector: the target embedding that will be compared with all embeddings.
        :param top_k: the top k embeddings selected.
        :return:
        """
        import numpy as np
        from scipy.spatial.distance import cdist

        embeddings = np.array(self.embeddings_list)  # [batch, 1536]

        # calculate cosine distance
        cosine_dists = cdist(np.array([vector]), embeddings, 'cosine').flatten()
        cosine_similarity = 1 - cosine_dists

        # get top k similarity index, topk_similarity = cosine_similarity[topk_indexes]
        topk_indexes = np.argpartition(cosine_similarity, -top_k)[-top_k:]

        data_num = len(embeddings)
        if top_k < data_num:
            return [{"embeddings": self.embeddings_list[index], "text": self.texts_list[index],
                     "similarity": cosine_similarity[index]} for index in topk_indexes]
        else:
            return self.load_all_data()

    def update(self, text: Optional[str] = None, values: Optional[List[float]] = None) -> Optional[bool]:
        vectors = self.load_all_data()
        try:
            with open(self.embedding_db, 'w') as jsonl_writer:
                for vector in vectors:
                    if vector['text'] == text:
                        vector['embedding'] = values
                    jsonl_writer.write(json.dumps(vector) + '\n')
        except Exception as e:
            logger.exception("updating vector failed", e)
            return False
        return True

    def upsert(self, vectors: Optional[List[Dict]] = None):
        """
        :param vectors: processor.upsert([{'embedding': [1.0, 2.0, 3.0], 'text': text, 'metadata': {'key': 'value'}},
                                          {'embedding': [4.0, 5.0, 6.0], 'text': text, 'metadata': {'key': 'value'}}])
        :return:
        """
        if vectors is None:
            raise ValueError(
                "input vectors can not be None, and it's format should be: [{'embedding': [1.0, 2.0, 3.0], 'text': text, 'metadata': {'key': 'value'}}].")

        # check if the text had been added
        texts = self.load_all_text()
        vectors_add = [vector for vector in vectors if vector["text"] not in set(texts)]

        with open(self.embedding_db, 'a', encoding="utf-8") as jsonl_reader:
            for vector in vectors_add:
                try:
                    jsonl_reader.write(json.dumps(vector) + '\n')
                except Exception as e:
                    logger.error(f"insert: {str(vector)} failed, " + e.__str__())

    def delete(self, text: Optional[str] = None, delete_all: Optional[bool] = False):
        # delete all
        if delete_all:
            try:
                open(self.embedding_db, 'w').close()
            except Exception as e:
                logger.error("delete vector failed, " + e.__str__())
                return False
            return True

        # do not delete all
        try:
            vectors = self.load_all_data()
            with open(self.embedding_db, 'w', encoding="utf-8") as jsonl_writer:
                for vector in vectors:
                    if vector['text'] != text:
                        jsonl_writer.write(json.dumps(vector) + '\n')
        except Exception as e:
            logger.error("delete vector failed, " + e.__str__())
            return False
        return True

    def load_all_data(self) -> Optional[List[Dict[str, Any]]]:
        vectors = []
        if not Path(self.embedding_db).is_file():
            logger.warning("embedding db is not a file, return None vectors!")
            return vectors

        # read all jsonl line
        with open(self.embedding_db, 'r') as jsonl_writer:
            for json_line in jsonl_writer:
                vectors.append(json.loads(json_line.strip()))
        return vectors

    def load_all_data_to_list(self):
        embeddings_list, texts_list = [], []
        if not Path(self.embedding_db).is_file():
            logger.warning("embedding db is not a file, return None vectors!")
            return embeddings_list, texts_list

        # read all jsonl line
        with open(self.embedding_db, 'r') as jsonl_writer:
            for json_line in jsonl_writer:
                dict_loads = json.loads(json_line.strip())
                embedding = dict_loads["embedding"]
                text = dict_loads["text"]

                # append
                embeddings_list.append(embedding)
                texts_list.append(text)
        return embeddings_list, texts_list

    def load_all_text(self):
        texts = []
        if not Path(self.embedding_db).is_file():
            logger.warning("embedding db is not a file, return None vectors!")
            return texts

        # read all jsonl line
        with open(self.embedding_db, 'r') as jsonl_reader:
            for json_line in jsonl_reader:
                texts.append(json.loads(json_line.strip())["text"])
        return texts
