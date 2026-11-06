from src.data.loader import build_dataloader
from src.config_loader import load_config
from src.models.baseline import MultimodalBaselineModel

def main():
    print("=== Khởi chạy toàn bộ Pipeline (Config -> Loader -> Model) ===")

    # 1. Đọc cấu hình
    try:
        config = load_config("configs/config.yaml")
        print("Đã tải cấu hình thành công:", config["model"]["name"])
    except Exception:
        print("Sử dụng cấu hình mặc định.")
        config = {"data": {"image_size": [224, 224], "batch_size": 2}, "model": {"embedding_dim": 512}}

    # 2. Chuẩn bị dữ liệu giả lập
    sample_data = [
        {"text": "Học máy nâng cao", "label": 0},
        {"text": "Xử lý ảnh số", "label": 1},
    ]

    # 3. Nạp DataLoader
    dataloader = build_dataloader(sample_data, batch_size=config["data"]["batch_size"])

    # 4. Khởi tạo mô hình
    model = MultimodalBaselineModel(embedding_dim=config["model"]["embedding_dim"])

    # 5. Chạy thử nghiệm qua các batch
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- Xử lý Batch {batch_idx + 1} ---")
        if "image" in batch and batch["image"].size > 0:
            print("Shape ảnh đầu vào:", batch["image"].shape)
            
            # Đưa qua mô hình lấy embedding
            embeddings = model.forward(batch["image"])
            print("Shape embedding đầu ra từ mô hình:", embeddings.shape)
            
        if "text" in batch:
            print("Văn bản đi kèm:", batch["text"])

if __name__ == "__main__":
    main()