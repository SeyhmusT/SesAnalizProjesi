import sys
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QLabel, QPushButton, QTextEdit, QMainWindow, QWidget, QHBoxLayout)
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt
import sounddevice as sd
import speech_recognition as sr
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
import librosa
import joblib

# Model and encoder paths
encoder_path = r"C:\Users\seyhmus\Desktop\SesAnalizProjesi\label_encoder.pkl"
model_path = r"C:\Users\seyhmus\Desktop\SesAnalizProjesi\model.pkl"

# Load model and encoder
model = joblib.load(model_path)
le = joblib.load(encoder_path)

# Speech-to-text function with speaker prediction
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Konuşmaya başlayabilirsiniz...")
        audio = recognizer.listen(source)
        print("Konuşma tamamlandı, metin dönüştürülüyor...")
        try:
            text = recognizer.recognize_google(audio, language="tr-TR")
            
            # Process audio for speaker prediction
            print("Ses işleniyor...")
            audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16).astype(np.float32) / 32768.0
            fs = 16000  # Default sampling rate for recognizer

            # MFCC and feature extraction
            n_mfcc = 13
            mfcc = librosa.feature.mfcc(y=audio_data, sr=fs, n_mfcc=n_mfcc)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            fm = np.mean(mfcc, axis=1, keepdims=True)
            fm_expanded = np.tile(fm, (1, mfcc.shape[1]))

            # Combine features
            combined_features = np.vstack((mfcc, delta_mfcc, delta2_mfcc, fm_expanded))
            X_new = np.mean(combined_features, axis=1).reshape(1, -1)  # Model input shape

            # Predict speaker
            label = model.predict(X_new)[0]
            prediction = le.inverse_transform([label])[0]

            return text, prediction
        except sr.UnknownValueError:
            return "Anlaşılamadı. Lütfen tekrar deneyin.", None
        except sr.RequestError:
            return "Servis hatası. Lütfen internet bağlantınızı kontrol edin.", None

# Emotion analysis function
def turkish_emotion_analysis(text):
    emotion_model = pipeline("text-classification", model="savasy/bert-base-turkish-sentiment-cased")
    results = emotion_model(text, top_k=3)
    emotions = {res['label']: res['score'] for res in results}
    total_score = sum(emotions.values())
    emotions_percentage = {emotion: round(score / total_score * 100, 2) for emotion, score in emotions.items()}
    return emotions_percentage

# Topic analysis function
def topic_analysis(text):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["spor", "teknoloji", "dünya", "sanat", "sağlık", "eğitim", "ekonomi", "politika"]
    result = classifier(text, candidate_labels)
    categories = result['labels']
    scores = result['scores']
    category_results = {categories[i]: round(scores[i] * 100, 2) for i in range(len(categories))}
    return category_results

# Ses histogrami ve spektrogram
def plot_audio_analysis(audio_data, sample_rate):
    plt.figure(figsize=(12, 8))

    # Plot waveform
    plt.subplot(3, 1, 1)
    plt.plot(audio_data, color='blue')
    plt.title("Ses Dalgası", fontsize=16)
    plt.xlabel("Zaman (örnek)", fontsize=12)
    plt.ylabel("Genlik", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot histogram
    plt.subplot(3, 1, 2)
    plt.hist(audio_data, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Ses Verisi Histogramı", fontsize=16)
    plt.xlabel("Genlik", fontsize=12)
    plt.ylabel("Frekans", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot spectrogram
    plt.subplot(3, 1, 3)
    plt.specgram(audio_data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title("Spektrogram", fontsize=16)
    plt.xlabel("Zaman (s)", fontsize=12)
    plt.ylabel("Frekans (Hz)", fontsize=12)
    plt.colorbar(label="Yoğunluk (dB)")

    plt.tight_layout()
    plt.show()

# PyQt5 Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Konuşma Analizi")
        self.setGeometry(200, 200, 600, 400)

        # Set dark theme
        self.set_dark_theme()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # Text display
        self.text_display = QTextEdit()
        self.text_display.setFont(QFont("Tahoma", 12))
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("background-color: #2E2E2E; color: white; padding: 10px; border: none;")
        main_layout.addWidget(self.text_display)

        # Buttons
        self.record_button = QPushButton("Konuşmayı Kaydet")
        self.record_button.setFont(QFont("Georgia", 10))
        self.record_button.setStyleSheet("background-color: #2E2E2E; color: white; padding: 10px; border-radius: 5px; border: 1px solid white;")
        self.record_button.clicked.connect(self.perform_analysis)
        button_layout.addWidget(self.record_button)

        self.clear_button = QPushButton("Temizle")
        self.clear_button.setFont(QFont("Georgia", 10))
        self.clear_button.setStyleSheet("background-color: #2E2E2E; color: white; padding: 10px; border-radius: 5px; border: 1px solid white;")
        self.clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_button)

        self.histogram_button = QPushButton("Histogram Göster")
        self.histogram_button.setFont(QFont("Georgia", 10))
        self.histogram_button.setStyleSheet("background-color: #2E2E2E; color: white; padding: 10px; border-radius: 5px; border: 1px solid white;")
        self.histogram_button.clicked.connect(self.show_audio_analysis)
        button_layout.addWidget(self.histogram_button)

        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

    def perform_analysis(self):
        text, prediction = speech_to_text()
        self.text_display.append("<b>Konuşma Metni:</b>")
        self.text_display.append(text)

        if prediction:
            self.text_display.append(f"<br><b>Tahmin Edilen Kişi:</b> {prediction}")

        # Word count
        word_count = len(text.split())
        self.text_display.append(f"<br><b>Kelime Sayısı:</b> {word_count}")

        # Topic analysis
        topics = topic_analysis(text)
        self.text_display.append("<br><b>Konu Analizi:</b>")
        for topic, percentage in topics.items():
            self.text_display.append(f"{topic.capitalize()}: %{percentage}")

        # Emotion analysis
        emotions = turkish_emotion_analysis(text)
        self.text_display.append("<br><b>Duygu Analizi:</b>")
        for emotion, percentage in emotions.items():
            self.text_display.append(f"{emotion.capitalize()}: %{percentage}")

    def clear_text(self):
        self.text_display.clear()

    def show_audio_analysis(self):
        duration = 5  # Record duration in seconds
        sample_rate = 44100  # Sample rate in Hz
        self.text_display.append("<br><b>Ses Verisi:</b> Histogram ve Spektrogram oluşturuluyor...")

        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
        sd.wait()

        # Flatten the audio data and plot analysis
        audio_data = audio_data.flatten()
        plot_audio_analysis(audio_data, sample_rate)

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())