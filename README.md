# CS_radiotherapy_project# CS Radiotherapy Project

Projekt symulacyjny poświęcony modelowaniu matematycznemu i uczeniu maszynowemu w kontekście radioterapii onkologicznej. Projekt bada dynamikę wzrostu guza oraz wpływ promieniowania przy użyciu równań różniczkowych (PDE/ODE) oraz nowoczesnych metod AI (PINN - Physics-Informed Neural Networks).

## 📂 Struktura Projektu

Główna logika symulacji znajduje się w katalogu `simulations/`. Każdy notatnik odpowiada za inny aspekt modelowania:

* **`sim_01_pde.ipynb`** – Symulacja 2D z wykorzystaniem Równań Różniczkowych Cząstkowych (PDE). Modeluje przestrzenną dyfuzję guza i wpływ dawki promieniowania.
* **`sim_02_ode.ipynb`** – Modelowanie za pomocą Równań Różniczkowych Zwyczajnych (ODE). Skupia się na ewolucji masy guza w czasie bez uwzględniania przestrzeni.
* **`sim_03_sensitivity.ipynb`** – Analiza wrażliwości parametrów modelu.
* **`sim_04_node.ipynb`** – Neural ODE (Neuronalne Równania Różniczkowe). Wykorzystanie sieci neuronowych do aproksymacji dynamiki układu.
* **`sim_05_asimilation.ipynb`** – Asymilacja danych (Data Assimilation). Dopasowywanie modelu do obserwacji.
* **`sim_06_PINN.ipynb`** – Physics-Informed Neural Networks. Hybrydowe podejście łączące wiedzę fizyczną (równania PDE) z uczeniem głębokim (PyTorch) do przewidywania rozwoju guza.

## 🚀 Jak uruchomić projekt (How to turn on)

Ponieważ plik `requirements.txt` w repozytorium jest pusty, poniżej znajduje się instrukcja instalacji brakujących bibliotek zidentyfikowanych w kodzie.

### Wymagania wstępne
* Python 3.8+
* Zalecane utworzenie wirtualnego środowiska.

### Krok 1: Instalacja zależności

W terminalu wykonaj następujące polecenia:

```bash
# 1. Utworzenie wirtualnego środowiska (opcjonalnie, ale zalecane)
python -m venv .venv

# 2. Aktywacja środowiska
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. Instalacja bibliotek
pip install numpy matplotlib scipy torch imageio jupyterlab