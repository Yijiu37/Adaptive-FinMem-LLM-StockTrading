import numpy as np
from typing import List, Union

# 使用HuggingFace本地嵌入模型
from sentence_transformers import SentenceTransformer


class OpenAILongerThanContextEmb:
    """
    使用HuggingFace本地嵌入模型替代OpenAI API
    支持多种sentence-transformers模型，无需API Key
    """

    def __init__(
            self,
            openai_api_key: Union[str, None] = None,
            embedding_model: str = "all-MiniLM-L6-v2",
            chunk_size: int = 5000,
            verbose: bool = False,
    ) -> None:
        """
        初始化本地嵌入模型

        Args:
            openai_api_key: 保留参数以兼容原有接口，本地模型不需要
            embedding_model: HuggingFace模型名称，推荐:
                - "all-MiniLM-L6-v2": 快速，384维，适合测试
                - "all-mpnet-base-v2": 高质量，768维，推荐生产使用
                - "paraphrase-multilingual-MiniLM-L12-v2": 多语言支持
            chunk_size: 保留参数以兼容原有接口
            verbose: 是否显示进度条
        """
        self.model_name = embedding_model
        self.chunk_size = chunk_size
        self.verbose = verbose

        # 加载本地模型（首次会自动下载到缓存）
        print(f"[Embedding] Loading local model: {embedding_model}")
        self.emb_model = SentenceTransformer(embedding_model)
        self._embedding_dim = self.emb_model.get_sentence_embedding_dimension()
        print(f"[Embedding] Model loaded successfully! Dimension: {self._embedding_dim}")

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        对文本列表进行嵌入

        Args:
            text: 单个文本字符串或文本列表

        Returns:
            嵌入向量列表
        """
        if isinstance(text, str):
            text = [text]

        # 使用本地模型编码
        embeddings = self.emb_model.encode(
            text,
            show_progress_bar=self.verbose,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        对文本进行嵌入，返回numpy数组

        Args:
            text: 单个文本字符串或文本列表

        Returns:
            float32类型的numpy数组
        """
        return np.array(self._emb(text)).astype("float32")

    def get_embedding_dimension(self) -> int:
        """
        返回嵌入向量的维度

        Returns:
            嵌入维度
        """
        return self._embedding_dim