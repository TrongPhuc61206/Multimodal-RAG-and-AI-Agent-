import numpy as np

class MultimodalBaselineModel:
    """Mô hình baseline kết hợp đặc trưng ảnh và văn bản đơn giản."""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        # Khởi tạo trọng số ngẫu nhiên cho mô hình giả lập
        self.W_image = np.random.randn(224 * 224 * 3, embedding_dim) * 0.01
        self.W_text = np.random.randn(100, embedding_dim) * 0.01

    def forward(self, image_batch: np.ndarray) -> np.ndarray:
        """Xử lý batch ảnh đầu vào và ánh xạ về không gian embedding."""
        if image_batch.size == 0:
            return np.array([])
        
        # Làm phẳng batch ảnh: [batch_size, H, W, C] -> [batch_size, H*W*C]
        batch_size = image_batch.shape[0]
        flat_images = image_batch.reshape(batch_size, -1)
        
        # Chiếu về không gian embedding (giả lập qua phép nhân ma trận thu gọn)
        # Lấy một phần trọng số cho khớp chiều dữ liệu test
        weight_subset = self.W_image[:flat_images.shape[1], :self.embedding_dim]
        embeddings = np.dot(flat_images, weight_subset)
        
        return embeddings