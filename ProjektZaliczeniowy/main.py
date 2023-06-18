import io

import pandas as pd
import sqlite3
import tkinter as tk
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import pickle
from pandasgui import show

import matplotlib.pyplot as plt
from tkinter import messagebox













wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
wine_data.columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
                'od280/od315_of_diluted_wines', 'proline']

#treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(wine_data.drop('class', axis=1), wine_data['class'], test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Ocena
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

#Zapisanie
with open('model.pkl','wb') as file:
    pickle.dump(model, file)

#Odczytanie
with open('model.pkl','rb') as file:
    loaded_model = pickle.load(file)

#GUI
root = tk.Tk()
root.title("Wine Classifier")

def train_model():
    global model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    messagebox.showinfo("Am i Training?", "Model Trained, i think ;-;")

def test_model():
    accuracy = model.score(X_test, y_test)
    messagebox.showinfo("TESTING_CAUTION", f"Accuracy is shown in here : {accuracy}")

def predict_new_data():# to be done
    pass


# Funkcja do ponownego budowania modelu
def rebuild_model():
    train_model()

# Tworzenie przycisków
button_font = ("Arial", 12, "bold")
train_button = tk.Button(root, text="Train like John Cena", command=train_model, bg="red", width=50, height=2, font=button_font)

train_button.pack()

test_button = tk.Button(root, text="Test accuracy", command=test_model, bg="Blue", width=25, height=2, font=button_font)
test_button.pack()

predict_button = tk.Button(root, text="NOT WORKING", command=predict_new_data, bg="red", width=13, height=2, font=button_font)
predict_button.pack()

rebuild_button = tk.Button(root, text="|Rebuild|", command=rebuild_model, bg="yellow", width=25, height=2, font=button_font)
rebuild_button.pack()

# Przeglądanie danych w tabelce
def browse_data():
    show(wine_data)

browse_button = tk.Button(root, text="|Browse|", command=browse_data, bg="green", width=30, height=2, font=button_font)
browse_button.pack()

# Wizualizacja danych na wykresie
def plot_data():
    plt.plot(wine_data['alcohol'], wine_data['color_intensity'], 'bo')
    plt.xlabel('Alcohol')
    plt.ylabel('Color Intensity')
    plt.show()

plot_button = tk.Button(root, text="Plot Data", command=plot_data, bg="red", width=20, height=2, font=button_font)
plot_button.pack()

# Przechowywanie danych w bazie SQLite
def save_to_database():
    conn = sqlite3.connect('wines.db')
    wine_data.to_sql('wine_table', conn, if_exists='replace', index=False)
    messagebox.showinfo("Database", "Data saved to SQLite database.")

save_button = tk.Button(root, text="Save to Database", command=save_to_database, bg="red", width=40, height=2, font=button_font)
save_button.pack()

# Odczytanie danych z bazy SQLite
def read_from_database():
    conn = sqlite3.connect('wines.db')
    query = "SELECT * FROM wine_table"
    result = pd.read_sql_query(query, conn)
    messagebox.showinfo("Database", "Data read from SQLite database.")
    show(result)

read_button = tk.Button(root, text="Read from Database", command=read_from_database, bg="red", width=25, height=2, font=button_font)
read_button.pack()



















root.mainloop()






