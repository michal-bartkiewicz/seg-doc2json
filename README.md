# hacknation2026-odwolanie2json

Narzędzie odczytuje polskie dokumenty związane z zamówieniami publicznymi i zapisuje wydobyte pola w JSON.

## Szybki start dla jury na VM

Umieść pliki PDF lub tekst UTF-8 w `/opt/solution/input`. Obraz Dockera z Pythonem 3.12 i pliki modelu w `/opt/solution/model_bundle` są przygotowane na VM. W katalogu `/opt/solution` uruchom polecenie z [RUN.md](RUN.md).

```bash
docker run --rm \
  --cpus=2 --memory=12g \
  -v /opt/solution/input:/input:ro \
  -v /opt/solution/output:/output \
  -v /opt/solution/model_bundle:/model_bundle:ro \
  hacknation2026-odwolanie2json:latest \
  --input /input \
  --output /output \
  --roberta-bundle /model_bundle \
  --roberta-config /opt/app/transfer-order-candidate-inference-config.json \
  --roberta-device cpu
```

- Pliki w katalogu wejściowym lub w dowolnych podkatalogach są klasyfikowane według treści za pomocą ważonych wyrażeń regularnych. Obsługiwane typy to odwołanie (`odwolanie`), pełnomocnictwo (`pelnomocnictwo`) i potwierdzenie przelewu (`przelew`). Jeżeli pierwszy katalog ścieżki nosi nazwę jednego z tych typów, program przyjmuje typ z katalogu.
- Wyniki trafiają do `/opt/solution/output`. Każdy dokument zachowuje względną ścieżkę z rozszerzeniem zmienionym na `.json`. Przykładowo `input/sprawa/a.pdf` daje `output/sprawa/a.json`. Dla dokumentu odrzuconego przez klasyfikator wynik ma postać `{}`. W katalogu wynikowym powstają też `_report.json` i `_validation.json`.
- Poppler odczytuje warstwę tekstową PDF i renderuje strony do OCR w Tesseract (`pol+eng`). Wyrażenia regularne wydobywają pola. Opcjonalny, dostrojony ModernBERT weryfikuje kandydatów lokalnie na CPU. Powyższe polecenie go włącza. Gotowe rozwiązanie działa bez zewnętrznych API, kluczy i pobierania plików podczas przetwarzania.

## Uruchomienie lokalne na Linuksie lub WSL

Skrypt `./setup.sh` obsługuje Debian i Ubuntu z Pythonem 3.13 lub nowszym. Tworzy środowisko `.venv` i podczas instalacji może pobierać zależności systemowe i pakiety Pythona. Umieść dokumenty w `data/documents/` i uruchom:

```bash
./setup.sh
PYTHONPATH=src .venv/bin/python -m odwolanie2json.main
```

Wyniki lokalne trafiają domyślnie do `outputs/`. Aby doinstalować zależności modelu CPU, użyj `WITH_LOCAL_ROBERTA=1 ./setup.sh`. Następnie przekaż do programu katalog z lokalnymi plikami modelu dla trzech typów:

```bash
PYTHONPATH=src .venv/bin/python -m odwolanie2json.main \
  --roberta-bundle /ścieżka/do/lokalnego/modelu \
  --roberta-device cpu
```

## Jak działa ekstrakcja

Poppler używa `pdftotext` do odczytu PDF i `pdftoppm` do renderowania stron, których warstwa tekstowa jest niewystarczająca. Po OCR reguły korygują typowe błędy rozpoznawania. Sprawdzają też pola potwierdzeń przelewu, granice ról w pełnomocnictwach oraz nagłówki i pierwsze zarzuty w odwołaniach. Do uruchomienia nie potrzeba etykiet ani pliku `expected.json`.

ModernBERT odczytuje pierwotny tekst OCR, dzięki czemu pozycje znaków pozostają zgodne ze źródłem. Może uzupełnić brakujące pole, gdy model wskaże fragment źródłowy i wartość ma poprawny format. W PDF-ach z zaznaczalnym tekstem polityka `--roberta-policy rank` może zmienić przypisanie kandydata do roli mocodawcy lub pełnomocnika, gdy cały wskazany fragment ma spójną sekwencję BIO. Przy niejednoznacznym wyniku program zachowuje wartość z reguł. Model wczytuje pliki z `local_files_only=True`.

W pełnomocnictwach model stosuje odrębne etykiety BIO dla mocodawcy, podpisującego i pełnomocnika. Każdy kandydat ma pozycję w źródle i jest powiązany z konkretnym elementem listy JSON. Ocena roli używa pierwszego wystąpienia nazwy, porównuje etykietę roli z etykietą `O` i uwzględnia próg wytrenowany dla typu dokumentu zgodnie z konfiguracją. Alternatywny kandydat na pełnomocnika może uzupełnić pole, jeżeli spełnia próg reguły i model jednoznacznie wskazuje rolę. Pełnomocnik powiązany z numerem PESEL zachowuje swoją rolę. Pola bez wytrenowanej etykiety obsługują reguły. Metoda `RegexExtractor.extract_with_candidates()` udostępnia dane kandydatów.

Pliki źródłowe, etykiety, wyniki, obrazy OCR i wagi modelu są wyłączone z Git. Repozytorium zajmuje poniżej 2 GB.

## Raporty

`_report.json` zawiera zagregowane liczniki decyzji i czas przetwarzania. `_validation.json` zawiera informacje o obecności danych w źródle, sygnałach klasyfikacji, numerach stron z OCR i liczbie decyzji dla każdego dokumentu. Pole `extraction_route` przyjmuje `ocr`, gdy Tesseract przetworzył co najmniej jedną stronę, `text` przy ekstrakcji z tekstu oraz `null` dla dokumentu odrzuconego.

## Wyniki pomiarów lokalnych

Wyniki porównują dokładne pary znormalizowanych ścieżek i wartości JSON. Pomiar dla skanów wykorzystuje zapisany tekst z Tesseract i obejmuje klasyfikację. Sześć skanów z dwóch spraw odłożono przed dostrajaniem reguł i treningiem modelu. Zbiór 30 skanów obejmuje także 24 skany używane podczas prac nad rozwiązaniem.

| Zbiór | Dokumenty | Przed | Po |
| --- | ---: | ---: | ---: |
| Oryginalne skany | 30 | 0.7872 | 0.9740 |
| Odłożone oryginalne skany | 6 | 0.8182 | 0.9552 |
| PDF-y z zaznaczalnym tekstem, same reguły | 180 | 0.9818 | 1.0000 |

Dla 30 skanów uzyskano 300 trafień, 7 fałszywych trafień i 9 pominięć. Wartości pochodziły z reguł. ModernBERT zaproponował kandydatów, a ostateczne przypisania pozostały bez zmian. Uruchomienie CLI na sześciu odłożonych PDF-ach z lokalnym OCR i włączonym modelem odtworzyło wynik 0.9552. Przetworzenie 15 stron zajęło 79,4 s na laptopie deweloperskim. Pomiary dotyczą publicznych dokumentów używanych do rozwoju i ich wewnętrznego podziału.

Ostatnie poprawki obsługi układu tekstu podniosły wynik z 0.9980 do 1.0000 na tych samych 180 PDF-ach (2032 trafienia, 0 fałszywych trafień i 0 pominięć). Ekstraktor czyta opisy bankowe z sąsiednich kolumn, pobiera kwoty z kolumny obok daty transakcji i zachowuje nazwy mocodawców rozbite na kilka wierszy.

Eksperyment z ModernBERT uwzględniającym OCR uzyskał łączne micro-F1 równe 0.7513 dla dokładnych granic fragmentów w predykcjach na odłożonych częściach podziału. Wariant kontrolny bez OCR uzyskał 0.7101. Późniejszy wariant `transfer-order-candidate` uzyskał odpowiednio 0.7486 dla odwołań, 0.7074 dla pełnomocnictw i 0.8133 dla przelewów. Wyniki pochodzą z osobnych konfiguracji i mierzą rozpoznawanie fragmentów tekstu. Progi poszczególnych typów służą do dekodowania BIO. Do strojenia akceptacji kandydatów służy micro-F1 końcowych pól JSON liczone na podstawie łącznych TP, FP i FN dla predykcji poza częścią treningową. Wszystkie warianty tej samej sprawy powinny trafić do tej samej części podziału, a zbiór testowy pozostać odłożony.

Skrypt [`evaluate_candidate_predictions.py`](experiments/roberta/evaluate_candidate_predictions.py) odczytuje gotowe predykcje z JSONL. Raportuje recall@k, maksymalny wynik przy dostępnym zbiorze kandydatów oraz micro-F1 końcowych pól. Dane wejściowe zawierają wydobyte wartości i pozostają lokalnie. Raport pomija identyfikatory i wartości.
