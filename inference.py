import openai
from loguru import logger
from typing import List, Optional, Dict, Any


class ChatGPT:
    API_TYPE = ""
    API_KEY = ""
    API_BASE = ""
    header = {
        'X-User-Id': "",
    }

    def __init__(self, engine="aide-gpt-4-turbo", embedding_engine='aide-text-embedding-ada-002-v2', temperature=0):
        """
        :param engine: "aide-gpt-35-turbo-16k-4k", "aide-gpt-4-turbo"
        """
        self.engine = engine
        self.embedding_engine = embedding_engine
        self.temperature = temperature

    def start_embedding(self, text: Optional[str] = None):
        # env
        openai.api_type = self.API_TYPE
        openai.api_base = self.API_BASE
        openai.api_version = "2023-05-15"
        openai.api_key = self.API_KEY

        # start embedding
        resp = openai.Embedding.create(
            engine=self.embedding_engine,
            input=text,
            headers=self.header
        )
        resp_embedding = resp["data"][0]["embedding"]
        return resp_embedding

    @staticmethod
    def build_prompt(user_text: Optional[str] = None,
                     system_text: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        messages = [
            {'role': 'system',
             'content': system_text if system_text else " "},
            {'role': 'user', 'content': user_text}
        ]
        return messages

    def chat(self,
             user_text: Optional[str] = None,
             system_text: Optional[str] = None) -> Optional[str]:
        # env
        openai.api_type = self.API_TYPE
        openai.api_base = self.API_BASE
        openai.api_version = '2023-07-01-preview'
        openai.api_key = self.API_KEY

        # start chat
        res_content = None
        try:
            if system_text is not None:
                messages = self.build_prompt(user_text, system_text)
            else:
                messages = self.build_prompt(user_text)

            response = openai.ChatCompletion.create(
                engine=self.engine,
                messages=messages,
                headers=self.header,
                temperature=self.temperature
            )

            res_content = response['choices'][0]['message']['content']
        except Exception as e:
            logger.exception(e)
        return res_content
