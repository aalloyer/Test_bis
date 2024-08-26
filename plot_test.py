import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Création de données aléatoires
x = np.random.randn(100)
y = x * 2 + np.random.randn(100)

# Création du plot
fig, ax = plt.subplots()
ax.scatter(x, y)

# Affichage du plot avec Streamlit
st.pyplot(fig)
