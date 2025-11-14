import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

# Imports do Plotly
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# Imports do ML
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Cria a aplicação
app = Flask(__name__)

# --- Configuração de Pastas ---
UPLOAD_FOLDER = os.path.join(app.root_path, 'datasets')
STATIC_FOLDER = os.path.join(app.root_path, 'static')
MODELS_FOLDER = os.path.join(app.root_path, 'models')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

# Caminhos
PROCESSED_FILE_PATH = os.path.join(UPLOAD_FOLDER, 'uploaded_data.csv')
MODEL_FAIXAS_PATH = os.path.join(MODELS_FOLDER, 'modelo_faixas.pkl')
MODEL_SUCESSO_PATH = os.path.join(MODELS_FOLDER, 'modelo_sucesso.pkl')
ENCODERS_PATH = os.path.join(MODELS_FOLDER, 'encoders.pkl')


# --- Rota 1: Página Inicial ---
@app.route('/')
def index():
    return render_template('index.html')


# --- Rota 2: Upload (redireciona para ML) ---
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Nenhum arquivo enviado", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "Nenhum arquivo selecionado", 400

    if file and file.filename.endswith('.csv'):
        try:
            file.save(PROCESSED_FILE_PATH)
            return redirect(url_for('dashboard'))  # Vai direto pro ML
        except Exception as e:
            return f"Erro ao salvar o arquivo: {e}", 500
    else:
        return "Formato de arquivo inválido. Por favor, envie um .csv", 400


# --- Rota 3.5: Configurar Treino (se quiser re-treinar) ---
@app.route('/configurar-treino')
def configurar_treino():
    if not os.path.exists(PROCESSED_FILE_PATH):
        return redirect(url_for('index'))
    # Deleta modelos antigos para forçar re-treino
    if os.path.exists(MODEL_FAIXAS_PATH):
        os.remove(MODEL_FAIXAS_PATH)
    if os.path.exists(MODEL_SUCESSO_PATH):
        os.remove(MODEL_SUCESSO_PATH)
    if os.path.exists(ENCODERS_PATH):
        os.remove(ENCODERS_PATH)
    return redirect(url_for('machine_learning'))


# --- Rota 3: Dashboard ---
@app.route('/dashboard')
def dashboard():
    if not os.path.exists(PROCESSED_FILE_PATH):
        return redirect(url_for('index'))
    
    try:
        df = pd.read_csv(PROCESSED_FILE_PATH)
        df_original_shape = df.shape
        
        # ETL
        dicionario_colunas = {
            'Name': 'Nome', 'Platform': 'Plataforma', 'Year_of_Release': 'Ano_de_Lancamento',
            'Genre': 'Genero', 'Publisher': 'Publicadora', 'NA_Sales': 'Vendas_NA',
            'EU_Sales': 'Vendas_EU', 'JP_Sales': 'Vendas_JP', 'Other_Sales': 'Vendas_Outras',
            'Global_Sales': 'Vendas_Globais', 'Critic_Score': 'Nota_Critica',
            'Critic_Count': 'Contagem_Critica', 'User_Score': 'Nota_Usuario',
            'User_Count': 'Contagem_Usuario', 'Developer': 'Desenvolvedora', 'Rating': 'Classificacao'
        }
        df_clean = df.rename(columns=dicionario_colunas)
        df_clean['Nota_Usuario'] = pd.to_numeric(df_clean['Nota_Usuario'], errors='coerce')
        
        colunas_essenciais_pt = [
            'Ano_de_Lancamento', 'Nota_Critica', 'Nota_Usuario',
            'Desenvolvedora', 'Classificacao', 'Genero', 'Vendas_Globais', 
            'Contagem_Critica', 'Contagem_Usuario'
        ]
        df_clean = df_clean.dropna(subset=colunas_essenciais_pt)
        df_clean['Ano_de_Lancamento'] = pd.to_numeric(df_clean['Ano_de_Lancamento'], errors='coerce')
        
        agg_rules = {
            'Vendas_Globais': 'sum', 'Vendas_NA': 'sum', 'Vendas_EU': 'sum',
            'Vendas_JP': 'sum', 'Vendas_Outras': 'sum', 'Contagem_Critica': 'sum',
            'Contagem_Usuario': 'sum', 'Nota_Critica': 'mean', 'Nota_Usuario': 'mean',
            'Ano_de_Lancamento': 'min', 'Genero': 'first', 'Classificacao': 'first',
            'Desenvolvedora': 'first', 'Publicadora': 'first',
            'Plataforma': lambda x: ', '.join(x.unique())
        }
        df_agg = df_clean.groupby('Nome').agg(agg_rules).reset_index()
        df_agg = df_agg.rename(columns={'Plataforma': 'Plataformas'})
        
        outliers_extremos = ['Wii Sports']
        df_agg = df_agg[~df_agg['Nome'].isin(outliers_extremos)]
        
        # Análises
        df_agg_shape = df_agg.shape
        etl_info = {
            "original_rows": df_original_shape[0], "original_cols": df_original_shape[1],
            "cleaned_rows": df_agg_shape[0], "cleaned_cols": df_agg_shape[1],
            "rows_removed": df_original_shape[0] - df_agg_shape[0]
        }
        
        df_head = df_agg.head(10)
        html_table = df_head.to_html(classes='table table-striped table-hover table-sm', index=False)
        
        df_top_vendas = df_agg.sort_values(by='Vendas_Globais', ascending=False).head(20)
        df_top_vendas = df_top_vendas[['Nome', 'Plataformas', 'Genero', 'Vendas_Globais']]
        html_top_vendas = df_top_vendas.to_html(classes='table table-striped table-hover table-sm', index=False)
        
        df_top_notas = df_agg.sort_values(by='Nota_Critica', ascending=False).head(20)
        df_top_notas = df_top_notas[['Nome', 'Plataformas', 'Nota_Critica', 'Vendas_Globais']]
        html_top_notas = df_top_notas.to_html(classes='table table-striped table-hover table-sm', index=False)
        
        # Gráficos (código dos 14 gráficos aqui - mantive igual)
        plot_htmls = {}
        
        # Gráfico 1
        genre_sales = df_agg.groupby('Genero')['Vendas_Globais'].sum().sort_values(ascending=False).head(10).reset_index()
        fig1 = px.bar(genre_sales, x='Genero', y='Vendas_Globais', title='Top 10 Gêneros por Vendas Globais',
                      labels={'Vendas_Globais': 'Vendas (Milhões)', 'Genero': 'Gênero'},
                      color='Vendas_Globais', color_continuous_scale='Viridis')
        fig1.update_layout(height=500)
        plot_htmls['genero_vendas'] = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 2
        fig2 = px.scatter(df_agg, x='Nota_Critica', y='Vendas_Globais',
                          hover_data=['Nome', 'Genero', 'Ano_de_Lancamento'],
                          title='Relação: Nota da Crítica vs. Vendas Globais',
                          labels={'Nota_Critica': 'Nota da Crítica', 'Vendas_Globais': 'Vendas (Milhões)'},
                          opacity=0.7, color='Genero', size='Contagem_Critica', size_max=15)
        fig2.update_layout(height=500)
        fig2.update_yaxes(range=[0, 30])
        fig2.update_traces(marker=dict(line=dict(width=0.5, color='white')))
        plot_htmls['scatter_critica'] = pio.to_html(fig2, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 3
        fig3 = px.scatter(df_agg, x='Nota_Usuario', y='Vendas_Globais',
                          hover_data=['Nome', 'Genero', 'Ano_de_Lancamento', 'Nota_Critica'],
                          title='Relação: Nota do Usuário vs. Vendas Globais',
                          labels={'Nota_Usuario': 'Nota do Usuário', 'Vendas_Globais': 'Vendas (Milhões)'},
                          opacity=0.7, color='Genero', size='Contagem_Usuario', size_max=30)
        fig3.update_layout(height=500)
        fig3.update_yaxes(range=[0, 30])
        fig3.update_traces(marker=dict(line=dict(width=0.5, color='white')))
        plot_htmls['scatter_usuario'] = pio.to_html(fig3, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 4
        fig4 = px.box(df_agg, x='Genero', y='Vendas_Globais',
                      title='Distribuição de Vendas por Gênero (Foco 0-5M)',
                      labels={'Vendas_Globais': 'Vendas (Milhões)', 'Genero': 'Gênero'}, color='Genero')
        fig4.update_layout(height=600)
        fig4.update_yaxes(range=[0, 5])
        plot_htmls['boxplot_genero'] = pio.to_html(fig4, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 5
        fig5 = px.box(df_agg, x='Classificacao', y='Vendas_Globais',
                      title='Distribuição de Vendas por Classificação (Foco 0-5M)',
                      labels={'Vendas_Globais': 'Vendas (Milhões)', 'Classificacao': 'Classificação ESRB'},
                      color='Classificacao')
        fig5.update_layout(height=500)
        fig5.update_yaxes(range=[0, 5])
        plot_htmls['boxplot_classificacao'] = pio.to_html(fig5, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 6
        fig6 = px.scatter(df_agg, x='Contagem_Critica', y='Vendas_Globais',
                          hover_data=['Nome', 'Genero', 'Nota_Critica', 'Nota_Usuario'],
                          title='Relação: "Hype" (Contagem Crítica) vs. Vendas',
                          labels={'Contagem_Critica': 'Nº de Críticos', 'Vendas_Globais': 'Vendas (Milhões)'},
                          opacity=0.7, color='Nota_Critica', size='Vendas_Globais', size_max=50,
                          color_continuous_scale='RdYlGn')
        fig6.update_layout(height=500)
        fig6.update_yaxes(range=[0, 30])
        fig6.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
        plot_htmls['scatter_hype'] = pio.to_html(fig6, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 7
        df_temporal = df_agg[df_agg['Ano_de_Lancamento'] >= 1996].groupby('Ano_de_Lancamento')['Vendas_Globais'].mean().reset_index()
        fig7 = px.line(df_temporal, x='Ano_de_Lancamento', y='Vendas_Globais', markers=True,
                       title='Média de Vendas Globais por Ano de Lançamento',
                       labels={'Ano_de_Lancamento': 'Ano', 'Vendas_Globais': 'Média de Vendas (Milhões)'})
        fig7.update_layout(height=500)
        plot_htmls['line_temporal'] = pio.to_html(fig7, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 8
        colunas_corr = ['Vendas_Globais', 'Nota_Critica', 'Nota_Usuario', 'Contagem_Critica', 'Contagem_Usuario']
        corr = df_agg[colunas_corr].corr()
        fig8 = px.imshow(corr, text_auto='.2f', aspect='auto',
                         title='Heatmap de Correlação (O Guia do ML)',
                         color_continuous_scale='RdBu_r', labels=dict(color="Correlação"))
        fig8.update_layout(height=600)
        plot_htmls['heatmap'] = pio.to_html(fig8, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 9
        df_temporal_count = df_agg[df_agg['Ano_de_Lancamento'] >= 1996].groupby('Ano_de_Lancamento').size().reset_index(name='Contagem')
        fig9 = px.line(df_temporal_count, x='Ano_de_Lancamento', y='Contagem', markers=True,
                       title='Contagem de Jogos Lançados por Ano',
                       labels={'Ano_de_Lancamento': 'Ano', 'Contagem': 'Número de Jogos'})
        fig9.update_traces(line_color='green')
        fig9.update_layout(height=500)
        plot_htmls['line_temporal_count'] = pio.to_html(fig9, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 10
        genre_count = df_agg['Genero'].value_counts().reset_index()
        genre_count.columns = ['Genero', 'Contagem']
        genre_count = genre_count.sort_values('Contagem', ascending=True)
        fig10 = px.bar(genre_count, x='Contagem', y='Genero', orientation='h',
                       title='Contagem de Jogos por Gênero (Todos)',
                       labels={'Contagem': 'Número de Jogos', 'Genero': 'Gênero'},
                       color='Contagem', color_continuous_scale='Spectral')
        fig10.update_layout(height=700)
        plot_htmls['bar_genre_count'] = pio.to_html(fig10, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 11
        df_regional = df_agg.groupby('Genero')[['Vendas_NA', 'Vendas_EU', 'Vendas_JP', 'Vendas_Outras']].sum().reset_index()
        df_regional_melted = df_regional.melt(id_vars='Genero', 
                                               value_vars=['Vendas_NA', 'Vendas_EU', 'Vendas_JP', 'Vendas_Outras'],
                                               var_name='Regiao', value_name='Vendas')
        df_regional_melted['Regiao'] = df_regional_melted['Regiao'].map({
            'Vendas_NA': 'América do Norte', 'Vendas_EU': 'Europa',
            'Vendas_JP': 'Japão', 'Vendas_Outras': 'Outras'
        })
        fig11 = px.bar(df_regional_melted, x='Genero', y='Vendas', color='Regiao',
                       title='Distribuição Regional de Vendas por Gênero',
                       labels={'Vendas': 'Vendas (Milhões)', 'Genero': 'Gênero'}, barmode='stack',
                       color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        fig11.update_layout(height=600)
        fig11.update_xaxes(tickangle=45)
        plot_htmls['bar_regional_genero'] = pio.to_html(fig11, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 12
        df_regional_pct = df_regional.copy()
        df_regional_pct['Total'] = df_regional_pct[['Vendas_NA', 'Vendas_EU', 'Vendas_JP', 'Vendas_Outras']].sum(axis=1)
        df_regional_pct['NA_%'] = (df_regional_pct['Vendas_NA'] / df_regional_pct['Total'] * 100).round(1)
        df_regional_pct['EU_%'] = (df_regional_pct['Vendas_EU'] / df_regional_pct['Total'] * 100).round(1)
        df_regional_pct['JP_%'] = (df_regional_pct['Vendas_JP'] / df_regional_pct['Total'] * 100).round(1)
        df_regional_pct['Outras_%'] = (df_regional_pct['Vendas_Outras'] / df_regional_pct['Total'] * 100).round(1)
        heatmap_data = df_regional_pct[['Genero', 'NA_%', 'EU_%', 'JP_%', 'Outras_%']].set_index('Genero')
        heatmap_data.columns = ['América do Norte', 'Europa', 'Japão', 'Outras']
        fig12 = px.imshow(heatmap_data, text_auto='.1f', aspect='auto',
                          title='Mapa de Calor: % de Vendas por Região e Gênero',
                          labels=dict(x="Região", y="Gênero", color="% Vendas"),
                          color_continuous_scale='YlOrRd')
        fig12.update_layout(height=600)
        plot_htmls['heatmap_regional'] = pio.to_html(fig12, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 13
        fig13 = px.scatter(df_agg, x='Vendas_NA', y='Vendas_JP',
                           hover_data=['Nome', 'Genero', 'Vendas_Globais'],
                           title='Comparação: Vendas América do Norte vs. Japão',
                           labels={'Vendas_NA': 'Vendas NA (Milhões)', 'Vendas_JP': 'Vendas JP (Milhões)'},
                           opacity=0.6, color='Genero', size='Vendas_Globais', size_max=40)
        fig13.update_layout(height=600)
        fig13.add_shape(type='line', x0=0, y0=0, x1=20, y1=20, 
                        line=dict(color='red', dash='dash'))
        plot_htmls['scatter_na_vs_jp'] = pio.to_html(fig13, full_html=False, include_plotlyjs='cdn')
        
        # Gráfico 14
        vendas_total_regiao = {
            'América do Norte': df_agg['Vendas_NA'].sum(),
            'Europa': df_agg['Vendas_EU'].sum(),
            'Japão': df_agg['Vendas_JP'].sum(),
            'Outras': df_agg['Vendas_Outras'].sum()
        }
        df_pizza = pd.DataFrame(list(vendas_total_regiao.items()), columns=['Regiao', 'Vendas'])
        fig14 = px.pie(df_pizza, values='Vendas', names='Regiao',
                       title='Distribuição Global de Vendas por Região (Total)', hole=0.4,
                       color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        fig14.update_traces(textposition='inside', textinfo='percent+label')
        fig14.update_layout(height=500)
        plot_htmls['pizza_regional'] = pio.to_html(fig14, full_html=False, include_plotlyjs='cdn')

        return render_template('dashboard.html', 
                               table=html_table, 
                               plot_htmls=plot_htmls,
                               etl_info=etl_info,
                               html_top_vendas=html_top_vendas,
                               html_top_notas=html_top_notas)

    except Exception as e:
        return f"Erro ao processar o arquivo no dashboard: {e}", 500


# --- Rota 4: Treinar Modelos CUSTOMIZADO (com parâmetros do form) ---
@app.route('/treinar-modelos-customizado', methods=['POST'])
def treinar_modelos():
    if not os.path.exists(PROCESSED_FILE_PATH):
        return redirect(url_for('index'))
    
    try:
        # RECEBER HIPERPARÂMETROS DO FORMULÁRIO
        rf_n_estimators = int(request.form.get('rf_n_estimators', 100))
        rf_max_depth = int(request.form.get('rf_max_depth', 15))
        dt_max_depth = int(request.form.get('dt_max_depth', 10))
        knn_n_neighbors = int(request.form.get('knn_n_neighbors', 5))
        svm_c = float(request.form.get('svm_c', 1.0))
        
        # ETL (mesmo código)
        df = pd.read_csv(PROCESSED_FILE_PATH)
        
        dicionario_colunas = {
            'Name': 'Nome', 'Platform': 'Plataforma', 'Year_of_Release': 'Ano_de_Lancamento',
            'Genre': 'Genero', 'Publisher': 'Publicadora', 'NA_Sales': 'Vendas_NA',
            'EU_Sales': 'Vendas_EU', 'JP_Sales': 'Vendas_JP', 'Other_Sales': 'Vendas_Outras',
            'Global_Sales': 'Vendas_Globais', 'Critic_Score': 'Nota_Critica',
            'Critic_Count': 'Contagem_Critica', 'User_Score': 'Nota_Usuario',
            'User_Count': 'Contagem_Usuario', 'Developer': 'Desenvolvedora', 'Rating': 'Classificacao'
        }
        df_clean = df.rename(columns=dicionario_colunas)
        df_clean['Nota_Usuario'] = pd.to_numeric(df_clean['Nota_Usuario'], errors='coerce')
        
        colunas_essenciais_pt = [
            'Ano_de_Lancamento', 'Nota_Critica', 'Nota_Usuario',
            'Desenvolvedora', 'Classificacao', 'Genero', 'Vendas_Globais', 
            'Contagem_Critica', 'Contagem_Usuario'
        ]
        df_clean = df_clean.dropna(subset=colunas_essenciais_pt)
        df_clean['Ano_de_Lancamento'] = pd.to_numeric(df_clean['Ano_de_Lancamento'], errors='coerce')
        
        agg_rules = {
            'Vendas_Globais': 'sum', 'Vendas_NA': 'sum', 'Vendas_EU': 'sum',
            'Vendas_JP': 'sum', 'Vendas_Outras': 'sum', 'Contagem_Critica': 'sum',
            'Contagem_Usuario': 'sum', 'Nota_Critica': 'mean', 'Nota_Usuario': 'mean',
            'Ano_de_Lancamento': 'min', 'Genero': 'first', 'Classificacao': 'first',
            'Desenvolvedora': 'first', 'Publicadora': 'first',
            'Plataforma': lambda x: ', '.join(x.unique())
        }
        df_agg = df_clean.groupby('Nome').agg(agg_rules).reset_index()
        
        outliers_extremos = ['Wii Sports']
        df_agg = df_agg[~df_agg['Nome'].isin(outliers_extremos)]
        
        # Targets
        df_agg['Faixa_Vendas'] = pd.cut(df_agg['Vendas_Globais'], 
                                         bins=[0, 0.5, 2, 10, 100], 
                                         labels=['Flop', 'Moderado', 'Sucesso', 'Blockbuster'])
        
        df_agg['Sucesso_Binario'] = ((df_agg['Vendas_Globais'] > 2) | (df_agg['Nota_Critica'] > 75)).astype(int)
        df_agg['Sucesso_Binario'] = df_agg['Sucesso_Binario'].map({0: 'Fracasso', 1: 'Sucesso'})
        
        # Features com contexto regional
        df_regional_avg = df_agg.groupby('Genero').agg({
            'Vendas_NA': 'mean', 'Vendas_EU': 'mean', 'Vendas_JP': 'mean', 
            'Vendas_Outras': 'mean', 'Vendas_Globais': 'mean'
        }).reset_index()
        
        df_regional_avg['NA_Pct'] = (df_regional_avg['Vendas_NA'] / df_regional_avg['Vendas_Globais'] * 100).round(2)
        df_regional_avg['EU_Pct'] = (df_regional_avg['Vendas_EU'] / df_regional_avg['Vendas_Globais'] * 100).round(2)
        df_regional_avg['JP_Pct'] = (df_regional_avg['Vendas_JP'] / df_regional_avg['Vendas_Globais'] * 100).round(2)
        df_regional_avg['Outras_Pct'] = (df_regional_avg['Vendas_Outras'] / df_regional_avg['Vendas_Globais'] * 100).round(2)
        
        df_ml = df_agg.merge(df_regional_avg[['Genero', 'NA_Pct', 'EU_Pct', 'JP_Pct', 'Outras_Pct']], 
                             on='Genero', how='left')
        
        le_genero = LabelEncoder()
        le_classificacao = LabelEncoder()
        
        df_ml['Genero_encoded'] = le_genero.fit_transform(df_ml['Genero'])
        df_ml['Classificacao_encoded'] = le_classificacao.fit_transform(df_ml['Classificacao'])
        
        features = ['Genero_encoded', 'Classificacao_encoded', 'Nota_Critica', 
                    'Nota_Usuario', 'Contagem_Critica', 'NA_Pct', 'EU_Pct', 'JP_Pct', 'Outras_Pct']
        
        X = df_ml[features]
        
        # Treinar modelos - Faixas (COM PARÂMETROS CUSTOMIZADOS)
        y_faixas = df_ml['Faixa_Vendas']
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y_faixas, test_size=0.2, random_state=42)
        
        modelos_faixas = {
            'Random Forest': RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=knn_n_neighbors),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        resultados_faixas = {}
        for nome, modelo in modelos_faixas.items():
            modelo.fit(X_train_f, y_train_f)
            y_pred = modelo.predict(X_test_f)
            acc = accuracy_score(y_test_f, y_pred)
            resultados_faixas[nome] = {'modelo': modelo, 'accuracy': acc, 'y_pred': y_pred}
        
        with open(MODEL_FAIXAS_PATH, 'wb') as f:
            pickle.dump(modelos_faixas, f)
        
        # Treinar modelos - Sucesso (COM PARÂMETROS CUSTOMIZADOS)
        y_sucesso = df_ml['Sucesso_Binario']
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_sucesso, test_size=0.2, random_state=42)
        
        modelos_sucesso = {
            'Random Forest': RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=knn_n_neighbors),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', C=svm_c, probability=True, random_state=42)
        }
        
        resultados_sucesso = {}
        for nome, modelo in modelos_sucesso.items():
            modelo.fit(X_train_s, y_train_s)
            y_pred = modelo.predict(X_test_s)
            acc = accuracy_score(y_test_s, y_pred)
            resultados_sucesso[nome] = {'modelo': modelo, 'accuracy': acc, 'y_pred': y_pred}
        
        with open(MODEL_SUCESSO_PATH, 'wb') as f:
            pickle.dump(modelos_sucesso, f)
        
        # Salvar encoders e contexto COM HIPERPARÂMETROS USADOS
        with open(ENCODERS_PATH, 'wb') as f:
            pickle.dump({
                'le_genero': le_genero,
                'le_classificacao': le_classificacao,
                'generos': list(le_genero.classes_),
                'classificacoes': list(le_classificacao.classes_),
                'regional_context': df_regional_avg[['Genero', 'NA_Pct', 'EU_Pct', 'JP_Pct', 'Outras_Pct']],
                'y_test_faixas': y_test_f,
                'y_test_sucesso': y_test_s,
                'resultados_faixas': resultados_faixas,
                'resultados_sucesso': resultados_sucesso,
                'hiperparametros': {  # NOVO: salvar params usados
                    'rf_n_estimators': rf_n_estimators,
                    'rf_max_depth': rf_max_depth,
                    'dt_max_depth': dt_max_depth,
                    'knn_n_neighbors': knn_n_neighbors,
                    'svm_c': svm_c
                }
            }, f)
        
        print(f"✅ Modelos treinados!")
        print(f"   Faixas: {', '.join([f'{k}: {v['accuracy']:.2%}' for k, v in resultados_faixas.items()])}")
        print(f"   Sucesso: {', '.join([f'{k}: {v['accuracy']:.2%}' for k, v in resultados_sucesso.items()])}")
        
        return redirect(url_for('machine_learning'))
    
    except Exception as e:
        return f"Erro ao treinar modelos: {e}", 500


# --- Rota 5: Página de Machine Learning ---
@app.route('/machine-learning')
def machine_learning():
    modelos_treinados = os.path.exists(MODEL_FAIXAS_PATH) and os.path.exists(MODEL_SUCESSO_PATH)
    
    if not modelos_treinados:
        return render_template('machine_learning.html', 
                               modelos_treinados=False,
                               generos=[], 
                               classificacoes=[])
    
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    
    # Gráficos de comparação
    resultados_f = encoders['resultados_faixas']
    nomes_f = list(resultados_f.keys())
    accs_f = [v['accuracy'] * 100 for v in resultados_f.values()]
    
    fig_comp_faixas = go.Figure(data=[
        go.Bar(x=nomes_f, y=accs_f, 
               text=[f"{a:.1f}%" for a in accs_f],
               textposition='outside',
               marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
    ])
    fig_comp_faixas.update_layout(
        title='Comparação de Modelos: Classificação de Faixas',
        xaxis_title='Modelo',
        yaxis_title='Accuracy (%)',
        yaxis_range=[0, 100],
        height=400
    )
    grafico_comp_faixas = pio.to_html(fig_comp_faixas, full_html=False, include_plotlyjs='cdn')
    
    resultados_s = encoders['resultados_sucesso']
    nomes_s = list(resultados_s.keys())
    accs_s = [v['accuracy'] * 100 for v in resultados_s.values()]
    
    fig_comp_sucesso = go.Figure(data=[
        go.Bar(x=nomes_s, y=accs_s,
               text=[f"{a:.1f}%" for a in accs_s],
               textposition='outside',
               marker_color=['#11998e', '#38ef7d', '#ee0979', '#ff6a00', '#a8edea'])
    ])
    fig_comp_sucesso.update_layout(
        title='Comparação de Modelos: Sucesso Binário',
        xaxis_title='Modelo',
        yaxis_title='Accuracy (%)',
        yaxis_range=[0, 100],
        height=400
    )
    grafico_comp_sucesso = pio.to_html(fig_comp_sucesso, full_html=False, include_plotlyjs='cdn')
    
    return render_template('machine_learning.html',
                           modelos_treinados=True,
                           generos=encoders['generos'],
                           classificacoes=encoders['classificacoes'],
                           modelos_faixas=nomes_f,
                           modelos_sucesso=nomes_s,
                           grafico_comp_faixas=grafico_comp_faixas,
                           grafico_comp_sucesso=grafico_comp_sucesso,
                           hiperparametros=encoders.get('hiperparametros', None))  # NOVO!


# --- Rota 6: Fazer Predição ---
@app.route('/prever', methods=['POST'])
def prever():
    try:
        genero = request.form['genero']
        classificacao = request.form['classificacao']
        nota_critica = float(request.form['nota_critica'])
        nota_usuario = float(request.form['nota_usuario'])
        contagem_critica = int(request.form['contagem_critica'])
        tipo_predicao = request.form['tipo_predicao']
        modelo_escolhido = request.form['modelo_escolhido']
        
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        
        genero_encoded = encoders['le_genero'].transform([genero])[0]
        classificacao_encoded = encoders['le_classificacao'].transform([classificacao])[0]
        
        regional_context = encoders['regional_context']
        genero_stats = regional_context[regional_context['Genero'] == genero].iloc[0]
        
        na_pct = genero_stats['NA_Pct']
        eu_pct = genero_stats['EU_Pct']
        jp_pct = genero_stats['JP_Pct']
        outras_pct = genero_stats['Outras_Pct']
        
        X_input = np.array([[genero_encoded, classificacao_encoded, nota_critica, 
                             nota_usuario, contagem_critica, na_pct, eu_pct, jp_pct, outras_pct]])
        
        if tipo_predicao == 'faixas':
            with open(MODEL_FAIXAS_PATH, 'rb') as f:
                modelos = pickle.load(f)
            
            modelo = modelos[modelo_escolhido]
            predicao = modelo.predict(X_input)[0]
            probabilidades = modelo.predict_proba(X_input)[0]
            confianca = max(probabilidades) * 100
            
            cores_faixas = {'Flop': 'danger', 'Moderado': 'warning', 'Sucesso': 'success', 'Blockbuster': 'primary'}
            icones_faixas = {'Flop': '😞', 'Moderado': '😐', 'Sucesso': '🎉', 'Blockbuster': '🚀'}
            
            resultado = {
                'tipo': 'Faixa de Vendas',
                'predicao': predicao,
                'confianca': f"{confianca:.1f}%",
                'cor': cores_faixas[predicao],
                'icone': icones_faixas[predicao],
                'modelo_usado': modelo_escolhido
            }
        else:
            with open(MODEL_SUCESSO_PATH, 'rb') as f:
                modelos = pickle.load(f)
            
            modelo = modelos[modelo_escolhido]
            predicao = modelo.predict(X_input)[0]
            probabilidades = modelo.predict_proba(X_input)[0]
            confianca = max(probabilidades) * 100
            
            cores_sucesso = {'Fracasso': 'danger', 'Sucesso': 'success'}
            icones_sucesso = {'Fracasso': '❌', 'Sucesso': '✅'}
            
            resultado = {
                'tipo': 'Sucesso/Fracasso',
                'predicao': predicao,
                'confianca': f"{confianca:.1f}%",
                'cor': cores_sucesso[predicao],
                'icone': icones_sucesso[predicao],
                'modelo_usado': modelo_escolhido
            }
        
        # Recarregar gráficos
        resultados_f = encoders['resultados_faixas']
        nomes_f = list(resultados_f.keys())
        accs_f = [v['accuracy'] * 100 for v in resultados_f.values()]
        
        fig_comp_faixas = go.Figure(data=[
            go.Bar(x=nomes_f, y=accs_f, 
                   text=[f"{a:.1f}%" for a in accs_f],
                   textposition='outside',
                   marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
        ])
        fig_comp_faixas.update_layout(
            title='Comparação de Modelos: Classificação de Faixas',
            xaxis_title='Modelo', yaxis_title='Accuracy (%)',
            yaxis_range=[0, 100], height=400
        )
        grafico_comp_faixas = pio.to_html(fig_comp_faixas, full_html=False, include_plotlyjs='cdn')
        
        resultados_s = encoders['resultados_sucesso']
        nomes_s = list(resultados_s.keys())
        accs_s = [v['accuracy'] * 100 for v in resultados_s.values()]
        
        fig_comp_sucesso = go.Figure(data=[
            go.Bar(x=nomes_s, y=accs_s,
                   text=[f"{a:.1f}%" for a in accs_s],
                   textposition='outside',
                   marker_color=['#11998e', '#38ef7d', '#ee0979', '#ff6a00', '#a8edea'])
        ])
        fig_comp_sucesso.update_layout(
            title='Comparação de Modelos: Sucesso Binário',
            xaxis_title='Modelo', yaxis_title='Accuracy (%)',
            yaxis_range=[0, 100], height=400
        )
        grafico_comp_sucesso = pio.to_html(fig_comp_sucesso, full_html=False, include_plotlyjs='cdn')
        
        return render_template('machine_learning.html',
                               modelos_treinados=True,
                               generos=encoders['generos'],
                               classificacoes=encoders['classificacoes'],
                               modelos_faixas=nomes_f,
                               modelos_sucesso=nomes_s,
                               grafico_comp_faixas=grafico_comp_faixas,
                               grafico_comp_sucesso=grafico_comp_sucesso,
                               hiperparametros=encoders.get('hiperparametros', None),  # NOVO!
                               resultado=resultado,
                               form_data=request.form)
    
    except Exception as e:
        return f"Erro ao fazer predição: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)