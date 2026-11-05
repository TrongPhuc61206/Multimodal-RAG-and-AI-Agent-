from src.data.loader import build_dataloader

def main():
    print("=== Đang khởi chạy pipeline nạp dữ liệu (NumPy) ===")

    sample_data = [
        {"image_path": "data/sample1.jpg", "text": "   Một chú mèo vàng   ", "label": 0},
        {"image_path": "data/sample2.jpg", "text": "Bức tranh phong cảnh ", "label": 1},
    ]

    print(f"Tổng số mẫu dữ liệu: {len(sample_data)}")

    try:
        dataloader = build_dataloader(sample_data, batch_size=2)
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n--- Batch {batch_idx + 1} ---")
            if "image" in batch and batch["image"].size > 0:
                print("NumPy Image batch shape:", batch["image"].shape)  # Ví dụ: [2, 224, 224, 3]
            if "text" in batch:
                print("Văn bản đã làm sạch:", batch["text"])
                
    except FileNotFoundError as e:
        print(f"\n[Lưu ý]: {e}")
        print("-> Hãy thêm ảnh mẫu vào thư mục `data/` để test đầy đủ pipeline!")

if __name__ == "__main__":
    main()