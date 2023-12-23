import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Membaca file CSV ke dalam dataframe
df = pd.read_csv('D:/SKRIPSI/Folder Baru/video_games_esrb_rating.csv')

# Preprocessing: One-Hot Encoding dan Feature-Target Split
X = df.drop(columns=['title', 'esrb_rating'])  # Menghilangkan kolom 'title' dan 'esrb_rating'
y = df['esrb_rating']

# Split data menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Penskalaan fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inisialisasi model SVM
svm_classifier = SVC(C=1.0, kernel='rbf', random_state=42)

# Pelatihan model SVM
svm_classifier.fit(X_train_scaled, y_train)

# Prediksi pada data uji
y_pred = svm_classifier.predict(X_test_scaled)

# Akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(
    svm_classifier,
    X_test_scaled,
    y_test,
    display_labels=y.unique(),
    cmap=plt.cm.Blues,
    normalize=None
)
disp.ax_.set_title("Confusion Matrix")

print("Confusion Matrix:")
print(disp.confusion_matrix)

plt.show()

# Laporan klasifikasi
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:\n", classification_rep)