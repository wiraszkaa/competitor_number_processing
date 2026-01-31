"""
Skrypt do trenowania klasyfikatora cyfr używającego SVM+HOG.
"""

from pathlib import Path
from competitor_number_processing.digit_classifier import (
    DigitClassifier,
    load_mnist_digits
)


def train_digit_classifier():
    """Trenuje klasyfikator cyfr na datasecie MNIST."""
    
    print("=" * 80)
    print("🎓 TRENOWANIE KLASYFIKATORA CYFR (SVM + HOG)")
    print("=" * 80)
    print()
    
    # Wczytaj dane MNIST
    images, labels = load_mnist_digits(max_samples_per_digit=1000)
    
    # Stwórz klasyfikator
    classifier = DigitClassifier(
        img_size=(32, 32),
        hog_win_size=(32, 32),
        hog_block_size=(16, 16),
        hog_block_stride=(8, 8),
        hog_cell_size=(8, 8),
        hog_nbins=9
    )
    
    # Trenuj
    stats = classifier.train(images, labels, test_size=0.2)
    
    # Zapisz model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "digit_classifier.pkl"
    classifier.save_model(model_path)
    
    print("\n" + "=" * 80)
    print("✅ TRENOWANIE ZAKOŃCZONE")
    print("=" * 80)
    print(f"Dokładność: {stats['accuracy'] * 100:.2f}%")
    print(f"Model zapisany: {model_path}")
    print()


if __name__ == "__main__":
    train_digit_classifier()
