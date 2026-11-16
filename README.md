# Titanic: previsão de sobrevivência (Kaggle)

Notebook para a competição [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) do Kaggle.

## O que o notebook faz

1. **Análise exploratória:** taxa de sobrevivência por sexo, classe e faixa de idade, e mapa dos valores ausentes.
2. **Engenharia de atributos:**
   - título extraído do nome (Mr, Mrs, Miss, Master, Raro), usado também para preencher a idade ausente pela mediana do grupo;
   - tamanho da família e indicador de quem viajava sozinho;
   - presença de cabine.
3. **Comparação de modelos** com validação cruzada estratificada de 5 folds, contra um baseline que usa só o sexo: Regressão Logística, SVM, Random Forest, Gradient Boosting e um ensemble com votação soft, todos dentro de um `Pipeline` do scikit-learn com pré-processamento (imputação, one-hot e padronização).
4. **Treino final** do ensemble e geração do `submission.csv`.

## Decisões técnicas

- O pré-processamento fica dentro do `Pipeline`, então é ajustado só nos dados de treino de cada fold. Isso evita vazamento de informação na validação.
- A mediana de idade por título é calculada só no conjunto de treino.
- A escolha do modelo usa a média da validação cruzada, não uma única divisão treino/validação.

## Histórico de versões no Kaggle

| Versão | Abordagem | Validação cruzada | Placar público do Kaggle |
|---|---|---|---|
| 2 | 14 atributos (incluindo deck e grupos de bilhete) + Gradient Boosting | 0,841 | 0,744 |
| 3 | 9 atributos + ensemble regularizado (LR + SVM + RF) | ver notebook | ver placar |

A versão 2 teve ótima validação cruzada e placar abaixo do baseline por sexo: overfitting causado por atributos muito específicos em um dataset de 891 linhas. A versão 3 simplifica os atributos e combina modelos de famílias diferentes para reduzir a variância.

## Como rodar

```bash
pip install pandas scikit-learn matplotlib jupyter
# coloque train.csv e test.csv da competição em data/
jupyter notebook titanic.ipynb
```

No Kaggle, o notebook encontra os dados da competição em `/kaggle/input`.

`make_notebook.py` gera o `titanic.ipynb` a partir do código.
