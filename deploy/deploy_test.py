import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle
import yaml

# @st.cache
model = pickle.load(open('model.pkl', 'rb'))
data_pipeline = pickle.load(open('data_pipeline.pkl', 'rb'))
df = pd.read_csv("../properties_v2.csv", sep=",")
# config = open("C:/Users/erycm/Documents/ciencia dos dados/awari/calculadora_aluguel/deploy/config/config.yml","r")
config = ''
with open('C:/Users/erycm/Documents/ciencia dos dados/awari/calculadora_aluguel/deploy/config/config.yml') as f:
  config = yaml.safe_load(f)

def main():
  st.title('Calculo de Aluguel - Curitiba')
  bairro = st.sidebar.selectbox("Bairro", df['neighbourhood'].unique())
  quartos = st.sidebar.selectbox("Quarto", [1,2,3,4,5,6,7])
  banheiros = st.sidebar.selectbox("Banheiros", [1,2,3,4,5])
  garagem = st.sidebar.selectbox("Garagem", [0,1,2,3,4,5])
  area = st.sidebar.number_input(
    label="Área do apto", min_value=10, max_value=500, value=70, step=25
  )
  calcular = st.sidebar.button('Calcular')

  if calcular:
    input_data = pd.DataFrame(
      [
        [
          area,
          quartos,
          banheiros,
          garagem,
          bairro,
        ]
      ],
      columns=config["model_input"]["numerical_features"]
      + config["model_input"]["categorical_features"],
    )
    normalized_input_data = data_pipeline.transform(input_data)
    preco = model.predict(normalized_input_data)[0]
    
    st.subheader(f"Preço estimado: R$ {preco:,.2f}")
    with open("memoria.csv", "a") as f:
        f.writelines(f"{bairro},{area},{quartos},{banheiros},{garagem},{preco}\n")




if __name__ == '__main__':
  main()