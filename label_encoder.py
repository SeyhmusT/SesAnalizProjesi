from sklearn.preprocessing import LabelEncoder
import joblib

# Etiketleri dönüştürme
y = ['onurmalay','osmantalha', 'seyhmustun', 'tugrulhan']  # Etiketlerinizin listesi

# LabelEncoder'ı eğitin
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# LabelEncoder'ı kaydedin
joblib.dump(le, r"C:\Users\seyhmus\Desktop\SesAnalizProjesi\label_encoder.pkl")

print("LabelEncoder kaydedildi.")