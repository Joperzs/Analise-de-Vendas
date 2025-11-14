# Análise de Vendas de Jogos com Machine Learning

Este é um projeto de aplicação web completa, desenvolvido em Python com **Flask**, **Pandas** e **Scikit-learn**. A aplicação permite ao usuário fazer o upload de um dataset de vendas de jogos, realizar uma limpeza e transformação (ETL) complexa, visualizar um dashboard interativo com 14 gráficos e, por fim, treinar, comparar e utilizar 5 modelos de Machine Learning para prever o sucesso de novos jogos.

A interface foi customizada com um tema "PS2-style" em CSS puro, sem o uso de frameworks como Bootstrap.

## 🧑‍💻 Autor(es)

  * [Seu Nome Aqui]
  * [Nome do Colega 1 (se houver)]
  * [Nome do Colega 2 (se houver)]

-----

## 📊 Dataset: Video Game Sales with Ratings

O dataset utilizado é uma versão estendida do "Video Game Sales" do VGChartz, enriquecido com dados de avaliação do Metacritic.

  * **Fonte Original:** [Kaggle - Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)
  * **Contexto:** O dataset original continha dados brutos de lançamentos por plataforma. Para uma análise justa, nosso processo de ETL agrega os dados por **jogo único**, somando vendas de diferentes plataformas.

### Campos Relevantes do Dataset

  * **Name:** Nome do jogo.
  * **Platform:** Plataforma de lançamento (ex: PS2, X360).
  * **Year\_of\_Release:** Ano de lançamento.
  * **Genre:** Gênero do jogo.
  * **Publisher:** Empresa que publicou o jogo.
  * **Global\_Sales:** Vendas globais (em milhões).
  * **Critic\_Score:** Pontuação agregada da crítica (Metacritic).
  * **Critic\_Count:** Número de críticos na pontuação.
  * **User\_Score:** Pontuação agregada dos usuários (Metacritic).
  * **User\_Count:** Número de usuários na pontuação.
  * **Rating:** Classificação ESRB (ex: E, M, T).

-----

## metodologias Centrais e Features do Projeto

O projeto é dividido em três grandes pilares: O processo de ETL, o Dashboard de Análise Visual e o pipeline de Machine Learning.

### 1\. Processo de ETL (Extract, Transform, Load)

Assim que o usuário faz o upload do `.csv`, o `app.py` executa um pipeline de ETL robusto para limpar e preparar os dados. Esta é a etapa mais crucial para a qualidade das análises.

1.  **Renomeação de Colunas:** Colunas são traduzidas para o português (ex: `Name` -\> `Nome`).
2.  **Limpeza de Dados Faltantes (NaN):** Linhas que não possuem dados essenciais para o ML (como `Nota_Critica`, `Nota_Usuario`, `Genero`, etc.) são removidas.
3.  **Unificação de Plataformas (Agregação):** Esta é a transformação principal. O dataset original trata "GTA V" no PS3 e "GTA V" no X360 como duas linhas separadas. Nossa aplicação agrega todas as linhas com o mesmo `Nome` em um **jogo único**.
      * **Vendas** (`Vendas_Globais`, `Vendas_NA`, etc.) são **somadas**.
      * **Notas** (`Nota_Critica`, `Nota_Usuario`) são calculadas pela **média**.
      * **Plataformas** são unidas em uma única string (ex: "PS3, X360, PC").
4.  **Remoção de Outlier (Wii Sports):** O jogo "Wii Sports" (82.9M de vendas) é identificado e removido. Por ter sido vendido em *bundle* com o console Wii, ele não representa um comportamento de mercado natural e distorce severamente as médias, correlações e, principalmente, o treinamento dos modelos de ML.

### 2\. Análise Visual (Dashboard)

O `dashboard.html` apresenta 14 gráficos interativos gerados com **Plotly Express**, todos renderizados no tema escuro (`template='plotly_dark'`) para se adequar ao CSS.

**Análises Principais:**

  * **Análise Geral de Vendas:**
      * Top 10 Gêneros por Vendas Totais (Gráfico de Barras).
      * Scatter Plot: Nota da Crítica vs. Vendas (mostra correlação positiva).
      * Scatter Plot: Nota do Usuário vs. Vendas (mostra correlação mais fraca).
      * Boxplots: Distribuição de Vendas por Gênero e por Classificação (focados no range 0-5M para ver a "cauda longa").
  * **Análise Temporal e de "Hype":**
      * Scatter Plot: "Hype" (Nº de Críticos) vs. Vendas.
      * Gráfico de Linha: Média de Vendas Globais por Ano.
      * Gráfico de Linha: Contagem de Jogos Lançados por Ano.
  * **Análise Regional (A mais profunda):**
      * Gráfico de Barras Empilhadas: Distribuição Regional (NA, EU, JP, Outras) por Gênero.
      * **Heatmap de % Regional:** Mostra a *dominância* de mercado (ex: RPGs dominando no Japão).
      * Gráfico de Pizza: Market share total por região.
      * Scatter Plot: Vendas NA vs. Vendas JP (mostra a clara divisão de gostos entre ocidente e oriente).
  * **Análise de Features (Guia do ML):**
      * **Heatmap de Correlação:** Mostra a correlação entre todas as features numéricas, servindo como guia para a escolha de features do ML.

### 3\. Pipeline de Machine Learning (Predição)

A página `machine_learning.html` é o coração do projeto. Ela permite ao usuário configurar, treinar e usar os modelos.

#### Engenharia de Features

Não usamos apenas os dados brutos. Criamos features que dão mais contexto ao modelo:

1.  **Features Padrão:** `Nota_Critica`, `Nota_Usuario`, `Contagem_Critica`.
2.  **Encoding:** `Genero` e `Classificacao` são transformados em números usando `LabelEncoder`.
3.  **Feature Avançada (Contexto Regional):** Em vez de apenas dizer ao modelo que um jogo é "Action", nós calculamos a **média de performance de vendas por região para aquele gênero**. O modelo recebe features como `NA_Pct` (ex: "Action" vende 40% na NA) e `JP_Pct` (ex: "Action" vende 15% no JP). Isso dá ao modelo um contexto de mercado crucial.

#### Alvos Preditivos (Targets)

Treinamos dois tipos de modelos para responder a duas perguntas diferentes:

1.  **Modelo 1 (Multi-classe): Classificação de Faixas**
      * **Pergunta:** "Qual será o nível de vendas deste jogo?"
      * **Classes:** `Flop` (0-0.5M), `Moderado` (0.5-2M), `Sucesso` (2-10M), `Blockbuster` (10M+).
2.  **Modelo 2 (Binário): Sucesso ou Fracasso**
      * **Pergunta:** "Este jogo será considerado um sucesso?"
      * **Classes:** `Sucesso` ou `Fracasso`, com base em uma regra de negócio: (`Vendas_Globais > 2M` OU `Nota_Critica > 75`).

#### Treinamento e Predição

1.  **Treinamento Customizado:** O usuário **não** usa um modelo pré-treinado. Ele **configura os hiperparâmetros** (ex: profundidade da árvore, número de vizinhos) e clica em "Treinar".
2.  **Modelos Comparados:** A aplicação treina **5 modelos** em paralelo para cada alvo:
      * `RandomForestClassifier`
      * `DecisionTreeClassifier`
      * `KNeighborsClassifier`
      * `LogisticRegression`
      * `SVC (Support Vector Machine)`
3.  **Comparação:** Após o treino, a página exibe gráficos de acurácia, comparando a performance de todos os modelos.
4.  **Predição:** O usuário pode então preencher um formulário com dados de um "novo jogo" e **escolher qual dos modelos treinados** ele quer usar para fazer a predição.
5.  **Persistência:** Os modelos treinados e os `encoders` são salvos na pasta `/models` usando `pickle`, permitindo que as predições sejam feitas sem re-treinamento a cada recarga da página.

-----

## 🚀 Como Executar o Projeto

Siga estes passos para rodar a aplicação localmente.

### Pré-requisitos

  * Python 3.7 ou superior
  * `pip` (gerenciador de pacotes do Python)

### 1\. Clonar o Repositório

```bash
git clone [URL_DO_SEU_REPOSITORIO]
cd [NOME_DO_SEU_REPOSITORIO]
```

### 2\. Criar um Ambiente Virtual (Recomendado)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Instalar as Dependências

Crie um arquivo chamado `requirements.txt` na raiz do projeto com o seguinte conteúdo:

**`requirements.txt`**

```
Flask
pandas
plotly
scikit-learn
numpy
```

Em seguida, instale os pacotes:

```bash
pip install -r requirements.txt
```

### 4\. Executar a Aplicação

```bash
python app.py
```

### 5\. Acessar no Navegador

Abra seu navegador e acesse: **`http://127.0.0.1:5000`**

-----

## 🛠️ Tecnologias Utilizadas

  * **Back-end:** Python, Flask
  * **Análise de Dados:** Pandas, Numpy
  * **Visualização:** Plotly Express
  * **Machine Learning:** Scikit-learn
  * **Front-end:** HTML5, CSS3 (Customizado)

-----

## 📁 Estrutura do Projeto

```
/projeto-final/
├── app.py                 # O servidor Flask principal (ETL, Rotas, ML, Plots)
├── datasets/              # Onde os .csv do usuário são salvos
│   └── uploaded_data.csv  # (Criado após o primeiro upload)
├── models/                # Onde os modelos .pkl são salvos
│   ├── encoders.pkl       # (Salva os encoders e o contexto regional)
│   ├── modelo_faixas.pkl  # (Modelos do Alvo 1)
│   └── modelo_sucesso.pkl # (Modelos do Alvo 2)
├── static/                # Arquivos de estilo
│   └── ps2_theme.css      # O tema customizado
├── templates/             # Arquivos HTML
│   ├── index.html         # Página de Upload
│   ├── dashboard.html     # Dashboard com 14 gráficos
│   └── machine_learning.html # Página de treino e predição
└── requirements.txt       # Lista de dependências
```