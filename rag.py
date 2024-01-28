import time
import pinecone
from utils.vectorizer import EmbeddingProcessor
from inference import ChatGPT


def test_rag():
    chatgpt = ChatGPT()

    prompt = "第81届金球奖举行，获得最佳剧情片在内的五项大奖，成为当晚的大赢家？"
    answer = chatgpt.chat(prompt)
    print(prompt)
    print(answer)

    # 对输入prompt进行embedding
    prompt_embedding = chatgpt.start_embedding(text=prompt)

    # 快速query，速度有所提升
    emb_processor = EmbeddingProcessor("./assets/test-vector-db.jsonl")

    # 未进行优化的query
    s1 = time.time()
    queried = emb_processor.query(prompt_embedding, top_k=2)
    e1 = time.time()
    for queried_text in queried:
        print("queried text: " + str(queried_text["text"]))
    print("normal query time: " + str(e1 - s1))

    # fast query
    s = time.time()
    queried = emb_processor.fast_query(prompt_embedding, top_k=2)
    e = time.time()
    for queried_text in queried:
        print("queried text: " + str(queried_text["text"]))
    print("fast query time: " + str(e - s))

    # 构建新的prompt
    contexts = [item["text"] for item in queried]
    system_instruction = "已知如下信息：\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"

    answer = chatgpt.chat(prompt, system_text=system_instruction)
    print(prompt)
    print(answer)


def test_pinecone():
    """
    由于网络问题，还未进行测试
    """
    chatgpt = ChatGPT()

    # 初始化pinecone，api_key可以从官网进行申请
    pinecone.init(
        api_key='',
        environment='gcp-starter'
    )

    # login db
    index = None
    try:
        index = pinecone.Index('test-vector-db')
    except Exception as e:
        print("connect to vector db failed.", e)

    if index is not None:
        print("connect to vector db success.")

    append_content = "第81届金球奖举行，《奥本海默》获得最佳剧情片在内的五项大奖，成为当晚的大赢家。"
    append_content_embeddings = chatgpt.start_embedding(text=append_content)

    # upsert to pinecone
    index.upsert([("q1", append_content_embeddings, {"data": append_content})])

    # 需要提问的prompt
    prompt = "第81届金球奖举行，获得最佳剧情片在内的五项大奖，成为当晚的大赢家？"
    prompt_embedding = chatgpt.start_embedding(text=prompt)

    # data format by query
    """
    {
      'matches': [{
        'id': 'q1',
        'metadata': {'data': '2022年卡塔尔世界杯的冠军是卡塔尔'},
        'score': 0.952013373,
        'values': []
      }],
      'namespace': ''
    }
    """
    queried = index.query(prompt_embedding, top_k=3, include_metadata=True)
    contexts = [item['metadata']['data'] for item in queried['matches']]
    system_instruction = "已知如下信息：\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"

    # 重新进行对话
    answer = chatgpt.chat(prompt, system_text=system_instruction)
    print(prompt)
    print(answer)


if __name__ == "__main__":
    test_rag()
