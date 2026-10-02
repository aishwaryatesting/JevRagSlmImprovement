class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, passages, top_k=3):
        if not passages:
            return []
        scores = self.model.predict([[query, passage] for passage in passages])
        ranked = sorted(zip(scores, passages), key=lambda x: float(x[0]), reverse=True)
        return [passage for _, passage in ranked[:top_k]]
