### Luźny logbook aby się nie pogubić.
***
01.02.2023
1. Wyrzuciłem z oryginalnych danych index 8679, bo jego lista steps była pusta
2. [`TODO`] Zastanowić się nad 'qos' gdzie w dwóch wierszach brakuje wartośći
3. [`TODO`] Zastanowić się nad constraints. Z 12k danych 116 wierszy ma 'cascade' reszta None
4. For the record. Za pomocą **transform.ipynb** można end-to-end wygenerować wypłaczony csv. W środku wywołuje on funkcje z **mapping_functions.py**, aby przemapować zagnieżdżone dictionaries w kolumnach na płaskie DataFrame'y. W kwestii steps wygenerowana csvka jest ograniczona tylko do pierwszego stepa (reszta discardowana + na razie nie ma wsparcia żeby to automatycznie i bezpiecznie przemapować). W razie wątpliwości co dzieje się w kodzie dopisałem docstringi do wszystkich funkcji mapujących.
5. [`TODO`] Po wypłaszczeniu powstaje 270 kolumn 
    - należy oczyścić kolumny niepotrzebne, a jest ich bardzo dużo (Np. jedna wartość) 
    - zastanowić się co z kolumnami gdzie mamy część danych jako nan.
***
02.02.2023
