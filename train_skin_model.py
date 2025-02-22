# 1. Import Necessary Libraries
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ========================================================
# 2. Load and Clean CSV Data
# ========================================================
train_csv_path = 'train.csv'
test_csv_path = 'test.csv'

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Add '.jpg' extension to md5hash
train_df['md5hash'] = train_df['md5hash'].astype(str) + '.jpg'
test_df['md5hash'] = test_df['md5hash'].astype(str) + '.jpg'

# Combine label and md5hash to form the file path (e.g. "acne/xyz.jpg")
train_df['file_path'] = train_df['label'] + '/' + train_df['md5hash']

# Replace ddi_scale with fitzpatrick_centaur
train_df['fitzpatrick_scale'] = train_df['fitzpatrick_centaur']
test_df['fitzpatrick_scale'] = test_df['fitzpatrick_centaur']
train_df.drop(columns=['ddi_scale'], inplace=True)
test_df.drop(columns=['ddi_scale'], inplace=True)

# Remove rows with wrongly labelled data
train_df = train_df[train_df['qc'] != '3 Wrongly labelled']
test_df = test_df[test_df['qc'] != '3 Wrongly labelled']

# Encode the label column numerically
label_encoder = LabelEncoder()
train_df['label_numerical'] = label_encoder.fit_transform(train_df['label'])

# Encode partition labels if needed
train_df['nine_partition_numerical'] = label_encoder.fit_transform(train_df['nine_partition_label'])
train_df['three_partition_numerical'] = label_encoder.fit_transform(train_df['three_partition_label'])

# Drop original label columns (optional)
train_df.drop(['label', 'three_partition_label', 'nine_partition_label'], axis=1, inplace=True)

print("Train DataFrame after cleaning:\n", train_df.head())

# ========================================================
# 3. Build the Dataset in Memory
# ========================================================
# We'll load each image, resize it to 150x150, and store it in a NumPy array.
base_image_dir = './train/train'
image_size = (150, 150)

all_images = []
all_labels = []

for idx, row in train_df.iterrows():
    file_path = os.path.join(base_image_dir, row['file_path'])
    
    # Load and resize image
    if os.path.exists(file_path):
        with Image.open(file_path) as img:
            img = img.resize(image_size)
            # Convert to RGB if not already
            img = img.convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            # Normalize pixel values to [0,1] (optional)
            img_array /= 255.0
            
            all_images.append(img_array)
            all_labels.append(row['label_numerical'])
    else:
        print(f"Warning: File not found {file_path}")

# Convert lists to NumPy arrays
X = np.array(all_images, dtype=np.float32)
y = np.array(all_labels, dtype=np.int32)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# ========================================================
# 4. Split into Training and Validation Sets
# ========================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================================================
# 5. Build a Simple CNN Model
# ========================================================
num_classes = len(np.unique(y))  # Number of distinct labels

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Because y is integer-labeled
    metrics=['accuracy']
)

model.summary()

# ========================================================
# 6. Train and Save the Model
# ========================================================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

model.save('skin_condition_model.h5')
print("Model saved as 'skin_condition_model.h5'")


# test_base_dir = './test/test'  # Adjust if your test images are in a different folder
# test_images   = []
# test_md5hash  = []

# for idx, row in test_df.iterrows():
#     file_path = os.path.join(test_base_dir, row['file_path'])
#     if os.path.exists(file_path):
#         with Image.open(file_path) as img:
#             img = img.resize(image_size)
#             img = img.convert('RGB')
#             img_array = np.array(img, dtype=np.float32)
#             img_array /= 255.0
#             test_images.append(img_array)
#             test_md5hash.append(row['md5hash'])
#     else:
#         print(f"Warning: Test file not found {file_path}")
#         test_images.append(np.zeros((150, 150, 3), dtype=np.float32))  # dummy
#         test_md5hash.append(row['md5hash'])

# X_test = np.array(test_images, dtype=np.float32)
# print("Test images shape:", X_test.shape)

# # 7b. Predict Classes
# pred_probs = model.predict(X_test)
# pred_class_indices = np.argmax(pred_probs, axis=1)

# # 7c. Convert Numeric Classes Back to Original Labels
# #    Because we used label_encoder on the training labels
# pred_labels = label_encoder.inverse_transform(pred_class_indices)

# # 7d. Create submission DataFrame: "md5hash,label"
# submission_df = pd.DataFrame({
#     'md5hash': test_md5hash,
#     'label': pred_labels
# })

# submission_df.to_csv('submission.csv', index=False)
# print("Saved predictions to submission.csv")