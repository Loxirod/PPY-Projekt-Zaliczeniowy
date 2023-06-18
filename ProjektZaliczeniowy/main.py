import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import sqlite3
import matplotlib.pyplot as plt

# Wczytanie danych dotyczących wina
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
data.columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                'od280/od315_of_diluted_wines', 'proline']

# Podział danych na zbiór treningowy i testowy
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Budowa modelu klasyfikacji
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Ocena modelu
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Zapisanie modelu do pliku
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Odczytanie modelu z pliku
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Graficzny interfejs użytkownika (GUI)
# Tutaj można wykorzystać biblioteki takie jak Tkinter, PyQt, wxPython do tworzenia GUI

# Przeglądanie danych w tabelce
data_table = pd.DataFrame(data)
print(data_table)

# Wizualizacja danych na wykresie
plt.plot(data['alcohol'], data['color_intensity'], 'bo')
plt.xlabel('Alcohol')
plt.ylabel('Color Intensity')
plt.show()

# Przechowywanie danych w bazie SQLite
conn = sqlite3.connect('wine_database.db')
data.to_sql('wine_table', conn, if_exists='replace', index=False)

# Odczytanie danych z bazy SQLite
query = "SELECT * FROM wine_table"
result = pd.read_sql_query(query, conn)
print(result)