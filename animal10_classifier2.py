import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback

# 設定儲存資料的資料夾
output_dir = "C:\\Users\\BSP\\Desktop\\training_logs"  # 調整保存路徑
os.makedirs(output_dir, exist_ok=True)

# 設定資料集路徑
dataset_path = "C:\\Users\\BSP\\Desktop\\raw-img"  # 根據您的路徑調整

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

# 載入模型或初始化模型
try:
    model = load_model(os.path.join(output_dir, 'best_model.h5'))
    print("載入已保存的最佳模型成功")
except:
    print("未找到已保存的最佳模型，初始化新模型")
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

# 設置新的學習率並重新編譯模型
learning_rate = 1e-5
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 自定義回調函數：每10個epoch保存訓練圖表和歷史紀錄
class SaveTrainingProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 每10個epoch保存訓練圖表
        if (epoch + 1) % 5 == 0:
            self.save_metrics(epoch)

    def save_metrics(self, epoch):
        # 儲存當前的訓練歷史
        history = self.model.history.history
        epochs_so_far = range(1, epoch + 2)
        
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_so_far, history['loss'], label='Training Loss')
        plt.plot(epochs_so_far, history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_so_far, history['accuracy'], label='Training Accuracy')
        plt.plot(epochs_so_far, history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.tight_layout()
        filename = os.path.join(output_dir, f"epoch_{epoch+1}_metrics.png")
        plt.savefig(filename)
        plt.close()
        print(f"Epoch {epoch+1}: 儲存視覺化圖表至 {filename}")

# 使用 ModelCheckpoint 保存最佳模型
checkpoint = ModelCheckpoint(
    filepath=os.path.join(output_dir, 'best_model.keras'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# 訓練回調
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 開始訓練
epochs = 100
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping, checkpoint, SaveTrainingProgress()]
)

# 儲存完整的訓練歷史紀錄為 JSON 文件
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
