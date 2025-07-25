import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Paths
TRAIN_CSV = 'dataset/sign_mnist_train.csv'
TEST_CSV = 'dataset/sign_mnist_test.csv'
MODEL_PATH = 'model/sign_model.h5'
IMG_SIZE = 28
EPOCHS = 15
BATCH_SIZE = 32

# 1. Load Data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)


# 2. Prepare Data
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Reshape and normalize
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0

# Remap labels to 0-23 (since 'J' is missing)
import numpy as np

# Get sorted unique labels
unique_labels = sorted(np.unique(y_train))
label_map = {label: idx for idx, label in enumerate(unique_labels)}

# Map the labels
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

num_classes = len(unique_labels)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Optional: Split train into train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 3. Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val)
)

# 5. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# 6. Save Model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
