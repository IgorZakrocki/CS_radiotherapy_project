# CS Radiotherapy Project

Projekt symulacyjny poświęcony matematycznemu modelowaniu oraz zastosowaniom uczenia maszynowego w radioterapii onkologicznej. Głównym celem jest analiza dynamiki wzrostu guza oraz optymalizacja dawki promieniowania z wykorzystaniem podejść klasycznych (modele oparte o równania różniczkowe) oraz metod głębokiego uczenia, w szczególności **Physics-Informed Neural Networks (PINN)**.

## Struktura projektu

Główna logika obliczeniowa i eksperymenty znajdują się w katalogu `simulations/`. Każdy notatnik odpowiada innemu wariantowi modelu lub technice analitycznej:

- **`sim_01_pde.ipynb`**  
  Symulacja 2D oparta o **równania różniczkowe cząstkowe (PDE)** – modeluje przestrzenną dyfuzję guza oraz wpływ dawki promieniowania na tkankę.

- **`sim_02_ode.ipynb`**  
  Modelowanie ewolucji masy guza w czasie z użyciem **równań różniczkowych zwyczajnych (ODE)**, bez komponentu przestrzennego.

- **`sim_03_sensitivity.ipynb`**  
  **Analiza wrażliwości (Sensitivity Analysis)** z użyciem biblioteki **SALib**, pozwalająca ocenić wpływ parametrów wejściowych na wyniki symulacji.

- **`sim_04_node.ipynb`**  
  Implementacja **Neural ODE** – wykorzystanie sieci neuronowych do aproksymacji dynamiki układu dynamicznego.

- **`sim_05_asimilation.ipynb`**  
  Moduł **asymilacji danych (Data Assimilation)** służący do dopasowywania modelu do danych obserwacyjnych.

- **`sim_06_PINN.ipynb`**  
  Zastosowanie **Physics-Informed Neural Networks (PINN)** – podejście hybrydowe łączące ograniczenia wynikające z równań fizycznych (PDE) z uczeniem sieci neuronowych w celu predykcji rozwoju nowotworu.

## Wymagania systemowe

- Python **3.8+**
- Rekomendowane środowisko: **Linux/macOS** lub **WSL** na Windows (ze względu na wsparcie dla `make`)

Zależności są zdefiniowane w pliku `requirements.txt` i obejmują m.in.:
- `numpy`, `scipy` – obliczenia numeryczne
- `torch` – uczenie maszynowe / deep learning
- `matplotlib`, `seaborn` – wizualizacja
- `SALib` – analiza wrażliwości

## Instalacja i konfiguracja

Projekt wspiera automatyzację instalacji poprzez `Makefile`.

### Metoda 1: automatyczna (zalecana)

Utworzenie środowiska wirtualnego (`.venv`) oraz instalacja zależności:

```bash
make install
```

### Metoda 2: ręczna

Jeśli `make` nie jest dostępne, instalację można wykonać manualnie:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchamianie symulacji

### Wykonanie wszystkich symulacji (tryb wsadowy)

Aby automatycznie uruchomić wszystkie notatniki Jupyter z katalogu `simulations/` i zapisać wyniki bezpośrednio w plikach `.ipynb`:

```bash
make run
```

### Uruchamianie interaktywne

Praca w trybie interaktywnym z użyciem Jupyter Lab:

```bash
jupyter lab
```

Następnie otwórz wybrany plik z katalogu `simulations/`.

## Czyszczenie wyników

Usunięcie wygenerowanych wyników z notatników (np. przed commitowaniem zmian):

```bash
make clean
```

## Licencja

Projekt jest udostępniany na warunkach licencji opisanej w pliku `LICENSE`.
