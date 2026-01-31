"""
Skrypt testowy do wykrywania i rozpoznawania numerów graczy na całym obrazie.
Wykorzystuje metodę MSER do detekcji cyfr i SVM+HOG do ich rozpoznawania.
"""

from pathlib import Path
import cv2
import random

from competitor_number_processing.detector import (
    NumberRegionDetector,
    NumberDetectionConfig,
)
from competitor_number_processing.digit_classifier import DigitClassifier


def test_number_detection_full_image():
    """Testuje detekcję i rozpoznawanie numerów bezpośrednio na całym obrazie."""
    
    print("=" * 80)
    print("🔍 TEST DETEKCJI I ROZPOZNAWANIA NUMERÓW (MSER + SVM+HOG)")
    print("=" * 80)
    
    # Sprawdź czy istnieje wytrenowany model
    model_path = Path("models/digit_classifier.pkl")
    use_classifier = model_path.exists()
    
    if use_classifier:
        print("✅ Znaleziono wytrenowany model klasyfikatora")
        classifier = DigitClassifier()
        classifier.load_model(model_path)
    else:
        print("⚠️  Brak wytrenowanego modelu - tylko detekcja bez rozpoznawania")
        print("   Uruchom: uv run python train_digit_classifier.py")
        classifier = None
    
    print()
    
    # Ścieżki - używamy preprocessed
    images_dir = Path("cache/preprocessed")
    output_dir = Path("cache/number_detection_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sprawdź czy folder istnieje
    if not images_dir.exists():
        print(f"❌ Folder nie istnieje: {images_dir}")
        return
    
    # Znajdź przetworzone obrazy
    image_files = list(images_dir.glob('*_final.png')) + list(images_dir.glob('*_final.jpg'))
    
    if not image_files:
        print(f"❌ Nie znaleziono obrazów w {images_dir}")
        return
    
    print(f"📂 Znaleziono {len(image_files)} przetworzonych obrazów")
    
    # Wybierz losowe próbki
    num_samples = min(10, len(image_files))
    samples = random.sample(image_files, num_samples)
    
    print(f"🎯 Testuję na {num_samples} losowych obrazach\n")
    
    # Inicjalizuj detektor numerów (bez detektora osób)
    number_config = NumberDetectionConfig(
        min_region_size=(20, 35),
        max_region_size=(150, 250),
        min_aspect_ratio=1.0,
        max_aspect_ratio=3.5,
        min_fill_ratio=0.25,
        max_fill_ratio=0.85,
        min_edge_density=0.08,
        use_mser=True,
        use_canny=False,
        use_hsv=False,
        use_adaptive=False,
        nms_iou_threshold=0.4,
        min_confidence=0.5,
        max_candidates_per_person=20,  # Więcej bo analizujemy cały obraz
        group_digits=False
    )
    number_detector = NumberRegionDetector(number_config)
    
    # Statystyki
    total_candidates = 0
    results_by_method = {'mser': 0}
    
    # Przetwarzaj każdy obraz
    for idx, image_path in enumerate(samples, 1):
        print(f"\n{'=' * 80}")
        print(f"📸 [{idx}/{num_samples}] {image_path.name}")
        print(f"{'=' * 80}")
        
        # Wczytaj obraz
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Nie można wczytać: {image_path}")
            continue
        
        print(f"   Rozmiar: {image.shape[1]}x{image.shape[0]}")
        
        # Wykryj kandydatów na numerach BEZPOŚREDNIO na całym obrazie
        print(f"   🔍 Wykrywanie regionów cyfr (MSER)...")
        candidates = number_detector.detect_candidates(image)
        print(f"   ✅ Znaleziono {len(candidates)} kandydatów")
        
        # Rozpoznaj cyfry jeśli mamy klasyfikator
        recognized_digits = []
        if candidates and classifier:
            print(f"   🤖 Rozpoznawanie cyfr (SVM+HOG)...")
            
            for candidate in candidates:
                # Wytnij region cyfry
                x, y, w, h = candidate.x, candidate.y, candidate.width, candidate.height
                digit_roi = image[y:y+h, x:x+w]
                
                if digit_roi.size > 0:
                    # Rozpoznaj cyfrę
                    result = classifier.predict(digit_roi)
                    recognized_digits.append((candidate, result))
        
        if candidates:
            # Statystyki metod
            methods_used = {}
            for c in candidates:
                methods_used[c.method] = methods_used.get(c.method, 0) + 1
                results_by_method[c.method] = results_by_method.get(c.method, 0) + 1
            
            for method, count in methods_used.items():
                print(f"      - {method}: {count}")
            
            # Top 10 najlepszych kandydatów z rozpoznanymi cyframi
            top_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)[:10]
            print(f"\n   ⭐ Top 10 kandydatów:")
            for i, c in enumerate(top_candidates, 1):
                # Znajdź rozpoznaną cyfrę dla tego kandydata
                digit_info = ""
                for cand, result in recognized_digits:
                    if cand == c:
                        digit_info = f" → CYFRA: {result.digit} (pewność: {result.confidence:.2f})"
                        break
                
                print(f"      {i}. {c.method} - detekcja: {c.confidence:.2f}{digit_info}")
                print(f"         Bbox: ({c.x}, {c.y}, {c.width}, {c.height})")
                print(f"         Aspect: {c.aspect_ratio:.2f}, Fill: {c.fill_ratio:.2f}, Edge: {c.edge_density:.2f}")
        
        total_candidates += len(candidates)
        
        # Wizualizacja z rozpoznanymi cyframi
        viz_image = image.copy()
        
        # Rysuj kandydatów z rozpoznanymi cyframi
        for candidate in candidates:
            x, y, w, h = candidate.x, candidate.y, candidate.width, candidate.height
            
            # Kolor ramki - różowy dla MSER
            color = (255, 0, 255)  # Magenta
            
            # Narysuj ramkę
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), color, 2)
            
            # Dodaj rozpoznaną cyfrę jeśli dostępna
            if classifier:
                for cand, result in recognized_digits:
                    if cand == candidate:
                        # Tekst z cyfrą
                        label = f"{result.digit}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.0
                        font_thickness = 2
                        
                        # Tło dla tekstu
                        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                        cv2.rectangle(viz_image, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
                        
                        # Tekst
                        cv2.putText(viz_image, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness)
                        break
        
        # Dodaj legendę
        legend_bg_color = (0, 0, 0)
        legend_height = 100 if classifier else 80
        overlay = viz_image.copy()
        cv2.rectangle(overlay, (0, 0), (450, legend_height), legend_bg_color, -1)
        viz_image = cv2.addWeighted(overlay, 0.7, viz_image, 0.3, 0)
        
        # Tekst legendy
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 25
        y_offset = 25
        
        title = "DETEKCJA I ROZPOZNAWANIE CYFR" if classifier else "DETEKCJA CYFR"
        cv2.putText(viz_image, title, (10, y_offset), font, font_scale, (255, 255, 255), font_thickness)
        y_offset += line_height
        
        # Różowe ramki = MSER
        cv2.rectangle(viz_image, (10, y_offset - 15), (30, y_offset + 5), (255, 0, 255), 2)
        info_text = f"MSER: {len(candidates)} kandydatow"
        cv2.putText(viz_image, info_text, (40, y_offset), font, 0.5, (255, 0, 255), 1)
        y_offset += line_height
        
        if classifier:
            cv2.putText(viz_image, "Cyfry rozpoznane przez SVM+HOG", (10, y_offset), font, 0.5, (255, 255, 255), 1)
        
        # Zapisz wynik
        output_path = output_dir / f"{image_path.stem}_numbers.jpg"
        cv2.imwrite(str(output_path), viz_image)
        print(f"\n   💾 Zapisano: {output_path.name}")
    
    # Podsumowanie
    print("\n" + "=" * 80)
    print("📊 PODSUMOWANIE")
    print("=" * 80)
    print(f"Przeanalizowane obrazy:      {num_samples}")
    print(f"Znalezione kandydaty:        {total_candidates}")
    
    if num_samples > 0:
        avg_candidates = total_candidates / num_samples
        print(f"Średnio kandydatów/obraz:    {avg_candidates:.1f}")
    
    print(f"\nWykrycia wg metod:")
    for method, count in sorted(results_by_method.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
        print(f"  - {method:10s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nWyniki zapisane w: {output_dir}")
    print("=" * 80)
    print("✅ Test zakończony!")


if __name__ == "__main__":
    test_number_detection_full_image()
