import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 設定資料集路徑
dataset_path = "C:\\Users\\BSP\\Desktop\\raw-img"  # 根據您的路徑調整

# 定義影像大小和批次大小
img_height, img_width = 224, 224  # EfficientNetB0 使用 224x224 的輸入大小
batch_size = 32

# 資料增強與預處理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 將 20% 的數據用於驗證
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

# 解凍後30層
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

# 訓練參數與回調
epochs = 1000
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

# 開始訓練
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping]
)

# 儲存模型
model.save('animal10_efficientnet_finetuned.h5')
print("模型已成功保存為 'animal10_efficientnet_finetuned.h5'")

# 可視化訓練和驗證損失
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 可視化訓練和驗證準確率
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.tight_layout()
plt.show()

# 顯示模型評估結果
train_loss, train_accuracy = model.evaluate(train_data)
val_loss, val_accuracy = model.evaluate(validation_data)
print("\n模型評估結果：")
print(f"訓練集損失: {train_loss:.4f}")
print(f"訓練集準確率: {train_accuracy:.4f}")
print(f"驗證集損失: {val_loss:.4f}")
print(f"驗證集準確率: {val_accuracy:.4f}")