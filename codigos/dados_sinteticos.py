# 0. BIBLIOTECAS

from copy import deepcopy
import random as rd
import numpy as np
import pandas as pd


# 1. CONSTANTES

# 1.1 Pastas
pasta_dados_originais = '../dados/originais'
pasta_dados_sinteticos = '../dados/sinteticos'


# 1.2 Atributos / Colunas
colunas_renomeadas = [
    'IdPassageiro',
    'Regiao',
    'SonoCriogenico',
    'Cabine',
    'Destino',
    'Idade',
    'VIP',
    'ServicoQuarto',
    'RestauranteGourmet',
    'PetShop',
    'Spa',
    'RealidadeVirtual',
    'Nome',
    'CelularRoubado'
]

colunas_gastos = [
    'ServicoQuarto',
    'RestauranteGourmet',
    'PetShop',
    'Spa',
    'RealidadeVirtual'
]


# 2. DATASETS

df_original = pd.read_csv(f'{pasta_dados_originais}/train.csv')
df_novo = deepcopy(df_original)
df_novo.columns = colunas_renomeadas

# 3. TRANSFORMAÇÕES

# 3.1 Dicionários de mapeamento
mapa_regiao = {
    'Earth': 'Sudeste',
    'Europa': 'Sul',
    'Mars': 'Nordeste'
}

mapa_destino = {
    'PSO J318.5-22': 'Melmac',
    '55 Cancri e': 'Vegeta',
    'TRAPPIST-1e': 'Valinor'
}

# 3.2 Variáveis categóricas

# Regiao
regiao_ausente = df_novo['Regiao'].isna()
df_novo['Regiao'] = df_novo['Regiao'].map(mapa_regiao)
df_novo.loc[regiao_ausente, 'Regiao'] = np.random.choice(
    ['Centro-Oeste', 'Norte'],
    size=regiao_ausente.sum(),
    p=[0.6, 0.4]
)

# Destino
df_novo['Destino'] = df_novo['Destino'].map(mapa_destino)

# Golpes Sofridos
golpes = ['Jogo do Bilu', 'Reptiliano do Pix', 'Imperador da Galáxia', 'Namorado(a) Alien']

for i in range(len(df_novo)):
    if df_novo.at[i, 'SonoCriogenico']:
        df_novo.at[i, 'GolpesSofridos'] = 'Nenhum'
        
    else:
        qtd_golpes = rd.choices(population=[0, 1, 2, 3, 4], weights=(50, 30, 10, 8, 2), k=1)[0]
        
        if qtd_golpes == 0:
            df_novo.at[i, 'GolpesSofridos'] = 'Nenhum'
        
        else:
            df_novo.at[i, 'GolpesSofridos'] = rd.sample(population=golpes, k=qtd_golpes)

# 3.3 Variáveis numéricas

# Gastos Originais
for coluna in colunas_gastos:
    df_novo[coluna] *= 15

# Gastos Novos
gastos_base = (
    df_novo['ServicoQuarto'].dropna() +
    df_novo['RestauranteGourmet'].dropna() +
    df_novo['PetShop'].dropna() +
    df_novo['Spa'].dropna() +
    df_novo['RealidadeVirtual'].dropna()
)

multiplicador_coach = np.random.lognormal(mean=-1.2, sigma=1.0, size=len(df_novo))
multiplicador_net = np.random.lognormal(mean=-1.4, sigma=0.6, size=len(df_novo))

multiplicador_coach = np.clip(multiplicador_coach, None, 3.0)

df_novo['CoachEspacial'] = gastos_base * multiplicador_coach
df_novo['Internet'] = gastos_base * multiplicador_net

df_novo.loc[df_novo['SonoCriogenico'] == True, ['CoachEspacial', 'Internet']] = 0

df_novo['CoachEspacial'] = df_novo['CoachEspacial'].round(2)
df_novo['Internet'] = df_novo['Internet'].round(2)
