"""
Skrypt testowy do wykrywania numerów graczy bezpośrednio na całym obrazie.
Wykorzystuje metodę MSER do detekcji cyfr.
"""

from pathlib import Path
import cv2
import random

from competitor_number_processing.detector import (
    NumberRegionDetector,
    NumberDetectionConfig,
)


def test_number_detection_full_image():
    """Testuje detekcję numerów bezpośrednio na całym obrazie."""
    
    print("=" * 80)
    print("🔍 TEST DETEKCJI NUMERÓW NA CAŁYM OBRAZIE")
    print("=" * 80)
    
    # Ścieżki - używamy preprocessed zamiast detections
    images_dir = Path("cache/preprocessed")
    output_dir = Path("cache/number_detection_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sprawdź czy folder istnieje
    if not images_dir.exists():
        print(f"❌ Folder nie istnieje: {images_dir}")
        return
    
    # Znajdź przetworzone obrazy
    image_files = list(images_dir.glob('*_enhanced.png')) + list(images_dir.glob('*_enhanced.jpg'))
    
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
    total_persons = 0
    results_by_method = {
        'mser': 0,
        'hsv': 0,
        'adaptive': 0,
        'canny': 0
    }
    
    # Przetwarzaj każde zdjęcie
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
        
        # Wykryj osoby
        print(f"   🔍 Wykrywanie osób...")
        detections = person_detector.detect(image)
        print(f"   ✅ Znaleziono {len(detections)} osób")
        
        if not detections:
            continue
        
        total_persons += len(detections)
        
        # Wyodrębnij regiony
        regions = person_detector.extract_regions(image, detections, with_padding=True)
        
        # Dla każdej osoby wykryj kandydatów na numery
        all_image_candidates = []
        
        for person_idx, (region, detection) in enumerate(zip(regions, detections)):
            print(f"\n   👤 Osoba {person_idx + 1}/{len(detections)}:")
            print(f"      Region: {region.shape[1]}x{region.shape[0]}")
            
            # Wykryj kandydatów używając wszystkich metod
            candidates = number_detector.detect_candidates(region)
            
            print(f"      🎯 Znaleziono {len(candidates)} kandydatów (max 4)")
            
            if candidates:
                # Statystyki metod
                methods_used = {}
                for c in candidates:
                    methods_used[c.method] = methods_used.get(c.method, 0) + 1
                    results_by_method[c.method] = results_by_method.get(c.method, 0) + 1
                
                for method, count in methods_used.items():
                    print(f"         - {method}: {count}")
                
                # Najlepszy kandydat
                best = max(candidates, key=lambda c: c.confidence)
                print(f"      ⭐ Najlepszy: {best.method} (pewność: {best.confidence:.2f})")
                print(f"         Bbox: {best.bbox}")
                print(f"         Aspect ratio: {best.aspect_ratio:.2f}")
                print(f"         Fill ratio: {best.fill_ratio:.2f}")
                print(f"         Edge density: {best.edge_density:.2f}")
            
            total_candidates += len(candidates)
            
            # Wizualizacja regionu z kandydatami
            if candidates:
                viz_region = number_detector.visualize_candidates(
                    region, 
                    candidates,
                    show_method=True,
                    show_confidence=True
                )
                
                # Zapisz region z wizualizacją
                region_output = output_dir / f"{image_path.stem}_person_{person_idx:02d}_candidates.jpg"
                cv2.imwrite(str(region_output), viz_region)
            
            # Przetłumacz współrzędne kandydatów na współrzędne całego obrazu
            for candidate in candidates:
                x_global = detection.x + candidate.x
                y_global = detection.y + candidate.y
                
                # Stwórz nowego kandydata z globalnymi współrzędnymi
                from competitor_number_processing.detector import NumberRegionCandidate
                global_candidate = NumberRegionCandidate(
                    x=x_global,
                    y=y_global,
                    width=candidate.width,
                    height=candidate.height,
                    confidence=candidate.confidence,
                    method=candidate.method,
                    aspect_ratio=candidate.aspect_ratio,
                    fill_ratio=candidate.fill_ratio,
                    edge_density=candidate.edge_density
                )
                all_image_candidates.append(global_candidate)
        
        # Wizualizacja całego obrazu z wszystkimi kandydatami
        if all_image_candidates:
            viz_full = image.copy()
            
            # Najpierw narysuj regiony osób (grube niebieskie ramki)
            for detection in detections:
                x, y, w, h = detection.bbox
                # Gruba niebieska ramka pokazująca region osoby wykryty przez HOG+SVM
                cv2.rectangle(viz_full, (x, y), (x + w, y + h), (255, 0, 0), 4)
            
            # Potem narysuj kandydatów NA TYM SAMYM OBRAZIE
            viz_full = number_detector.visualize_candidates(
                viz_full,
                all_image_candidates,
                show_method=True,
                show_confidence=True
            )
            
            # Dodaj legendę w górnym lewym rogu
            legend_bg_color = (0, 0, 0)
            legend_height = 120
            overlay = viz_full.copy()
            cv2.rectangle(overlay, (0, 0), (400, legend_height), legend_bg_color, -1)
            viz_full = cv2.addWeighted(overlay, 0.7, viz_full, 0.3, 0)
            
            # Tekst legendy
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            line_height = 30
            y_offset = 25
            
            cv2.putText(viz_full, "LEGENDA:", (10, y_offset), font, font_scale, (255, 255, 255), font_thickness)
            y_offset += line_height
            
            # Niebieskie ramki = regiony osób
            cv2.rectangle(viz_full, (10, y_offset - 15), (30, y_offset + 5), (255, 0, 0), 3)
            cv2.putText(viz_full, "Regiony osob (HOG+SVM)", (40, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            # Zielone = canny
            cv2.rectangle(viz_full, (10, y_offset - 15), (30, y_offset + 5), (0, 255, 0), 2)
            cv2.putText(viz_full, "Kandydaci: Canny", (40, y_offset), font, 0.5, (0, 255, 0), 1)
            y_offset += line_height
            
            # Różowe = MSER
            cv2.rectangle(viz_full, (10, y_offset - 15), (30, y_offset + 5), (255, 0, 255), 2)
            cv2.putText(viz_full, "Kandydaci: MSER", (40, y_offset), font, 0.5, (255, 0, 255), 1)
            
            full_output = output_dir / f"{image_path.stem}_all_candidates.jpg"
            cv2.imwrite(str(full_output), viz_full)
            print(f"\n   💾 Zapisano: {full_output.name}")
            print(f"      ℹ️  Grube niebieskie ramki = regiony osób (HOG+SVM)")
            print(f"      ℹ️  Małe kolorowe ramki = kandydaci na cyfry (TYLKO w regionach osób)")
    
    # Podsumowanie
    print("\n" + "=" * 80)
    print("📊 PODSUMOWANIE")
    print("=" * 80)
    print(f"Przeanalizowane zdjęcia:     {num_samples}")
    print(f"Wykryte osoby:               {total_persons}")
    print(f"Znalezione kandydaty:        {total_candidates}")
    
    if total_persons > 0:
        avg_candidates = total_candidates / total_persons
        print(f"Średnio kandydatów/osobę:    {avg_candidates:.1f}")
    
    print(f"\nWykrycia wg metod:")
    for method, count in sorted(results_by_method.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
        print(f"  - {method:10s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nWyniki zapisane w: {output_dir}")
    print("=" * 80)
    print("✅ Test zakończony!")


if __name__ == "__main__":
    test_number_detection_full_image()
