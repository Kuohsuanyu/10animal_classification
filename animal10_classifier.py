import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
import os
import json

# 設定資料集路徑
dataset_path = "C:\\Users\\BSP\\Desktop\\raw-img"  # 根據您的路徑調整
output_dir = "training_logs"  # 記錄和圖片保存的資料夾
os.makedirs(output_dir, exist_ok=True)

# 定義影像大小和批次大小
img_height, img_width = 224, 224
batch_size = 32

# 資料增強與預處理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 使用 EfficientNetB0 作為基礎模型
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
for layer in base_model.layers[-30:]:
    layer.trainable = True

# 添加分類層
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 假設分類數為10
])

# 編譯模型
learning_rate = 1e-5
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 自定義回調函數：每10個epoch保存視覺化結果
class VisualizationCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            # 儲存目前的損失和準確率圖表
            self.plot_metrics(epoch, logs)

    def plot_metrics(self, epoch, logs):
        epochs_so_far = range(1, epoch + 2)
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_so_far, history.history['loss'], label='Training Loss')
        plt.plot(epochs_so_far, history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_so_far, history.history['accuracy'], label='Training Accuracy')
        plt.plot(epochs_so_far, history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.tight_layout()
        filename = os.path.join(output_dir, f"epoch_{epoch+1}_metrics.png")
        plt.savefig(filename)
        plt.close()
        print(f"Epoch {epoch+1}: 儲存視覺化圖表至 {filename}")

# 使用 ModelCheckpoint 每次模型改進後保存最佳模型
checkpoint = ModelCheckpoint(
    filepath=os.path.join(output_dir, 'best_model.keras'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# 進行訓練時進行動態學習率調整和早停
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

# 開始訓練並記錄每個epoch的歷史數據
epochs = 1000
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping, checkpoint, VisualizationCallback()]
)

# 儲存最終模型和訓練歷史
final_model_path = os.path.join(output_dir, 'animal10_efficientnet_finetuned.h5')
model.save(final_model_path)
print(f"模型已成功保存為 {final_model_path}")

# 將訓練歷史紀錄保存為JSON
history_path = os.path.join(output_dir, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history.history, f)
print(f"訓練歷史紀錄已保存為 {history_path}")

# 最終可視化訓練和驗證損失及準確率
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "final_training_results.png"))
plt.show()
print("最終訓練和驗證結果已保存並顯示。")

# 顯示最終評估結果
train_loss, train_accuracy = model.evaluate(train_data)
val_loss, val_accuracy = model.evaluate(validation_data)
print("\n模型最終評估結果：")
print(f"訓練集損失: {train_loss:.4f}")
print(f"訓練集準確率: {train_accuracy:.4f}")
print(f"驗證集損失: {val_loss:.4f}")
print(f"驗證集準確率: {val_accuracy:.4f}")
