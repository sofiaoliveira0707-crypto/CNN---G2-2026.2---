# Segurança pública inteligente em São Paulo: classificação de ocorrências com MLP

Global Solution - Aprendizado Profundo com Redes Neurais Convolucionais 

**Equipe:**
- Nícolas Borges Natal - 568230

- Pedro Ricardo de Almeida - 567056

- ⁠Sofia Fernandes de Lima Oliveira - 567824

## Aplicação web
🔗 [COLOQUE AQUI O LINK DO STREAMLIT]

## Sobre o projeto
MLP que classifica uma ocorrência já registrada como **roubo** ou **furto** na cidade de São Paulo.
O modelo não prevê se um crime vai acontecer: ele estima, para uma ocorrência com determinadas
características, a probabilidade de ser roubo ou furto.

## Dados
- Fonte: SSP-SP, Portal de Estatísticas, base "Celulares Subtraídos" (2023): https://www.ssp.sp.gov.br/estatistica/consultas
- Recorte: ocorrências na capital (campo CIDADE), rubricas de roubo e furto, sem boletins duplicados (91.878 registros); amostra estratificada de 20.000.
- Atributos de entrada: bairro, tipo de local, período do dia e dia da semana.

## Modelo
MLP escolhida: 1 camada oculta de 16 neurônios (ReLU) e saída sigmoid. Adam (lr 0,001), binary crossentropy, batch 64, Early Stopping.
Três arquiteturas foram comparadas no notebook `GS_2026_2_CNN.ipynb`.

## Como executar
1. Instalar o Python 3.12.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## Arquivos
- `app.py`: aplicação Streamlit
- `modelo_mlp.keras`: modelo treinado
- `encoder.joblib`: OneHotEncoder ajustado no treino
- `GS_2026_2_CNN.ipynb`: notebook com preparação, treino e avaliação
