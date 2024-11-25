import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1. Tải và chuẩn bị dữ liệu
train_dir = "./data/train"
val_dir = "./data/train"  # Sử dụng chung train để thử nghiệm ban đầu
test_dir = "./data/test"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Chuẩn hóa pixel về [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",  # 0: Cat, 1: Dog
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
)

# 2. Xây dựng mô hình
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),  # Giảm overfitting
    Dense(1, activation="sigmoid"),  # Phân loại nhị phân
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 3. Hàm dự đoán
def predict_image(img_path, model, is_trained=False):
    """
    Dự đoán ảnh trước hoặc sau khi huấn luyện.
    """
    try:
        img = image.load_img(img_path, target_size=(150, 150))  # Resize ảnh
        img_array = image.img_to_array(img) / 255.0  # Chuẩn hóa pixel
        img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

        if is_trained:
            prediction = model.predict(img_array)[0][0]  # Kết quả dự đoán
            if prediction > 0.5:
                print(f"Ảnh {os.path.basename(img_path)}: Chó")
            else:
                print(f"Ảnh {os.path.basename(img_path)}: Mèo")
        else:
            # Trường hợp chưa huấn luyện
            print(f"Ảnh {os.path.basename(img_path)}: Mô hình chưa được huấn luyện, dự đoán ngẫu nhiên.")
            random_label = np.random.choice(["Mèo", "Chó"])
            print(f"Ảnh {os.path.basename(img_path)}: Dự đoán ngẫu nhiên là {random_label}")

    except Exception as e:
        print(f"Lỗi khi dự đoán ảnh {img_path}: {e}")


# 4. Dự đoán trước khi huấn luyện
print("\nDự đoán trước khi huấn luyện:")
test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith(".jpg")]
for img_path in test_images:
    predict_image(img_path, model, is_trained=False)

# 5. Huấn luyện mô hình
print("\nBắt đầu huấn luyện...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
)

# 6. Dự đoán sau khi huấn luyện
print("\nDự đoán sau khi huấn luyện:")
for img_path in test_images:
    predict_image(img_path, model, is_trained=True)
