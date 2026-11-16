# Titanic: previsão de sobrevivência (Kaggle)

Notebook para a competição [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) do Kaggle.

## O que o notebook faz

1. **Análise exploratória:** taxa de sobrevivência por sexo, classe e faixa de idade, e mapa dos valores ausentes.
2. **Engenharia de atributos:**
   - título extraído do nome (Mr, Mrs, Miss, Master, Raro), usado também para preencher a idade ausente pela mediana do grupo;
   - tamanho da família e indicador de quem viajava sozinho;
   - presença de cabine e deck;
   - frequência do bilhete e tarifa por pessoa (grupos que viajaram com o mesmo bilhete).
3. **Comparação de modelos** com validação cruzada estratificada de 5 folds: Regressão Logística, SVM, Random Forest e Gradient Boosting, todos dentro de um `Pipeline` do scikit-learn com pré-processamento (imputação, one-hot e padronização).
4. **Treino final** do melhor modelo e geração do `submission.csv`.

## Decisões técnicas

- O pré-processamento fica dentro do `Pipeline`, então é ajustado só nos dados de treino de cada fold. Isso evita vazamento de informação na validação.
- A mediana de idade por título é calculada só no conjunto de treino.
- A escolha do modelo usa a média da validação cruzada, não uma única divisão treino/validação.

## Como rodar

```bash
pip install pandas scikit-learn matplotlib jupyter
# coloque train.csv e test.csv da competição em data/
jupyter notebook titanic.ipynb
```

No Kaggle, o notebook encontra os dados da competição em `/kaggle/input`.

`make_notebook.py` gera o `titanic.ipynb` a partir do código.
