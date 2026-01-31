"""
Klasyfikator cyfr używający SVM na cechach HOG.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import pickle
from dataclasses import dataclass


@dataclass
class DigitClassificationResult:
    """Wynik klasyfikacji cyfry."""
    digit: int  # Rozpoznana cyfra (0-9)
    confidence: float  # Pewność klasyfikacji
    probabilities: np.ndarray  # Prawdopodobieństwa dla wszystkich klas


class DigitClassifier:
    """
    Klasyfikator cyfr używający SVM na cechach HOG.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 32),
        hog_win_size: Tuple[int, int] = (32, 32),
        hog_block_size: Tuple[int, int] = (16, 16),
        hog_block_stride: Tuple[int, int] = (8, 8),
        hog_cell_size: Tuple[int, int] = (8, 8),
        hog_nbins: int = 9
    ):
        """
        Inicjalizuje klasyfikator.
        
        Args:
            img_size: Rozmiar obrazu do którego skalujemy cyfry
            hog_win_size: Rozmiar okna dla HOG
            hog_block_size: Rozmiar bloku dla HOG
            hog_block_stride: Krok bloku dla HOG
            hog_cell_size: Rozmiar komórki dla HOG
            hog_nbins: Liczba binów histogramu dla HOG
        """
        self.img_size = img_size
        
        # Konfiguracja HOG
        self.hog = cv2.HOGDescriptor(
            _winSize=hog_win_size,
            _blockSize=hog_block_size,
            _blockStride=hog_block_stride,
            _cellSize=hog_cell_size,
            _nbins=hog_nbins
        )
        
        # SVM - będzie wytrenowany później
        self.svm = None
        self.trained = False
    
    def preprocess_digit(self, digit_image: np.ndarray) -> np.ndarray:
        """
        Przygotowuje obraz cyfry do ekstrakcji cech.
        
        Args:
            digit_image: Obraz cyfry (BGR lub grayscale)
            
        Returns:
            Przetworzony obraz gotowy do HOG
        """
        # Konwersja do grayscale jeśli kolorowy
        if len(digit_image.shape) == 3:
            gray = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = digit_image.copy()
        
        # Skaluj do standardowego rozmiaru
        resized = cv2.resize(gray, self.img_size, interpolation=cv2.INTER_AREA)
        
        # Wyrównaj histogram
        equalized = cv2.equalizeHist(resized)
        
        # Normalizacja 0-255
        normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def extract_features(self, digit_image: np.ndarray) -> np.ndarray:
        """
        Ekstraktuje cechy HOG z obrazu cyfry.
        
        Args:
            digit_image: Obraz cyfry (już przetworzony)
            
        Returns:
            Wektor cech HOG
        """
        # Oblicz HOG
        features = self.hog.compute(digit_image)
        
        # Spłaszcz do 1D
        return features.flatten()
    
    def train(
        self, 
        images: List[np.ndarray], 
        labels: List[int],
        test_size: float = 0.2
    ) -> dict:
        """
        Trenuje klasyfikator SVM na podanych danych.
        
        Args:
            images: Lista obrazów cyfr
            labels: Lista etykiet (0-9)
            test_size: Procent danych na zbiór testowy
            
        Returns:
            Statystyki treningu
        """
        print(f"🔧 Trenowanie klasyfikatora na {len(images)} przykładach...")
        
        # Przygotuj dane
        X = []
        y = []
        
        for img, label in zip(images, labels):
            # Przetwórz i ekstraktuj cechy
            processed = self.preprocess_digit(img)
            features = self.extract_features(processed)
            X.append(features)
            y.append(label)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print(f"   Wymiar cech: {X.shape[1]}")
        
        # Podziel na train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Konfiguracja SVM
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)  # Liniowy kernel - szybszy
        self.svm.setC(2.5)
        self.svm.setGamma(0.5)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6))
        
        # Trenuj
        print("   🏋️ Trenowanie SVM...")
        self.svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        self.trained = True
        
        # Testuj
        print("   📊 Testowanie...")
        _, y_pred = self.svm.predict(X_test)
        y_pred = y_pred.flatten().astype(int)
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Dokładność: {accuracy * 100:.2f}%\n")
        print("Raport klasyfikacji:")
        print(classification_report(y_test, y_pred, digits=3))
        
        # Macierz pomyłek
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
    
    def predict(self, digit_image: np.ndarray) -> DigitClassificationResult:
        """
        Rozpoznaje cyfrę na obrazie.
        
        Args:
            digit_image: Obraz cyfry
            
        Returns:
            Wynik klasyfikacji
        """
        if not self.trained or self.svm is None:
            raise RuntimeError("Klasyfikator nie został wytrenowany! Użyj train() lub load_model()")
        
        # Przetwórz i ekstraktuj cechy
        processed = self.preprocess_digit(digit_image)
        features = self.extract_features(processed)
        features = features.reshape(1, -1).astype(np.float32)
        
        # Predykcja
        _, result = self.svm.predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
        
        # Wynik - dla SVM to są odległości od hiperpłaszczyzn
        # Dla multi-class SVM (one-vs-one) mamy wyniki dla każdej pary klas
        raw_scores = result[0]
        
        # Najprostsza predykcja - najbliższa klasa
        _, prediction = self.svm.predict(features)
        digit = int(prediction[0][0])
        
        # Oblicz "prawdopodobieństwa" używając softmax na odległościach
        # (to jest aproksymacja, prawdziwe SVM nie daje prawdopodobieństw)
        if len(raw_scores) >= 10:
            # One-vs-rest: mamy score dla każdej klasy
            scores = raw_scores[:10]
        else:
            # One-vs-one: uproszczona konwersja
            scores = np.zeros(10)
            scores[digit] = 1.0
        
        # Softmax dla "prawdopodobieństw"
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        confidence = float(probabilities[digit])
        
        return DigitClassificationResult(
            digit=digit,
            confidence=confidence,
            probabilities=probabilities
        )
    
    def predict_batch(self, digit_images: List[np.ndarray]) -> List[DigitClassificationResult]:
        """
        Rozpoznaje wiele cyfr naraz.
        
        Args:
            digit_images: Lista obrazów cyfr
            
        Returns:
            Lista wyników klasyfikacji
        """
        results = []
        for img in digit_images:
            result = self.predict(img)
            results.append(result)
        return results
    
    def save_model(self, filepath: Path):
        """
        Zapisuje wytrenowany model do pliku.
        
        Args:
            filepath: Ścieżka do pliku
        """
        if not self.trained or self.svm is None:
            raise RuntimeError("Brak wytrenowanego modelu do zapisania")
        
        # Zapisz SVM
        svm_path = str(filepath).replace('.pkl', '_svm.xml')
        self.svm.save(svm_path)
        
        # Zapisz konfigurację
        config = {
            'img_size': self.img_size,
            'trained': self.trained,
            'svm_path': svm_path
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"💾 Model zapisany: {filepath}")
    
    def load_model(self, filepath: Path):
        """
        Wczytuje wytrenowany model z pliku.
        
        Args:
            filepath: Ścieżka do pliku
        """
        # Wczytaj konfigurację
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        self.img_size = config['img_size']
        self.trained = config['trained']
        
        # Wczytaj SVM
        svm_path = config['svm_path']
        self.svm = cv2.ml.SVM_load(svm_path)
        
        print(f"📂 Model wczytany: {filepath}")


def load_mnist_digits(
    max_samples_per_digit: int = 1000
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Wczytuje cyfry z datasetu MNIST używając sklearn.
    
    Args:
        max_samples_per_digit: Maksymalna liczba przykładów na cyfrę
        
    Returns:
        (images, labels)
    """
    print("📥 Wczytywanie datasetu MNIST...")
    
    from sklearn.datasets import load_digits
    
    # Load digits dataset (8x8 obrazy cyfr 0-9)
    digits = load_digits()
    
    images = []
    labels = []
    
    # Dla każdej cyfry wybierz losowe próbki
    for digit in range(10):
        # Znajdź wszystkie przykłady tej cyfry
        indices = np.where(digits.target == digit)[0]
        
        # Losuj maksymalnie max_samples_per_digit
        if len(indices) > max_samples_per_digit:
            indices = np.random.choice(indices, max_samples_per_digit, replace=False)
        
        for idx in indices:
            # Pobierz obraz (8x8) i przeskaluj do większego rozmiaru
            img = digits.images[idx]
            img = (img * 255 / 16).astype(np.uint8)  # Skaluj 0-16 -> 0-255
            
            # Powiększ do 32x32
            img_resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
            
            images.append(img_resized)
            labels.append(digit)
    
    print(f"   ✅ Wczytano {len(images)} obrazów cyfr")
    print(f"   Cyfry 0-9, ~{max_samples_per_digit} przykładów każdej")
    
    return images, labels
