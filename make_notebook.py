"""Gera o notebook titanic.ipynb (rode: python make_notebook.py)."""
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Titanic: previsão de sobrevivência com engenharia de atributos e validação cruzada

**Autor:** Breno Borges · Engenharia de Computação (UCDB)

Objetivo: prever quem sobreviveu ao naufrágio do Titanic a partir dos dados dos passageiros.

Etapas:
1. Análise exploratória rápida
2. Engenharia de atributos (título do nome, tamanho da família, cabine, faixas de tarifa)
3. Comparação de modelos com validação cruzada estratificada (5 folds)
4. Treino do melhor modelo e geração do arquivo de submissão"""),
    code("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

SEED = 42
DATA_DIR = '/kaggle/input/titanic' if os.path.exists('/kaggle/input/titanic') else 'data'
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
print(train.shape, test.shape)
train.head()"""),
    md("## 1. Análise exploratória"),
    code("""print('Taxa geral de sobrevivência:', round(train.Survived.mean(), 3))
print('\\nValores ausentes no treino:')
print(train.isna().sum()[lambda s: s > 0])
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
train.groupby('Sex').Survived.mean().plot.bar(ax=ax[0], title='Sobrevivência por sexo', rot=0)
train.groupby('Pclass').Survived.mean().plot.bar(ax=ax[1], title='Sobrevivência por classe', rot=0)
train.groupby(pd.cut(train.Age, [0, 12, 18, 30, 50, 80])).Survived.mean().plot.bar(ax=ax[2], title='Sobrevivência por idade', rot=0)
plt.tight_layout(); plt.show()"""),
    md("""Sexo e classe são os sinais mais fortes ("mulheres e crianças primeiro" e a localização das cabines da 1ª classe). A idade tem muitos valores ausentes, então vale preenchê-la usando o título do nome (Master, Miss, Mrs, Mr), que já carrega informação de idade."""),
    md("## 2. Engenharia de atributos"),
    code("""def criar_atributos(df):
    df = df.copy()
    df['Title'] = df.Name.str.extract(r',\\s*([^\\.]+)\\.', expand=False).str.strip()
    df['Title'] = df.Title.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df.loc[~df.Title.isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Title'] = 'Raro'
    df['FamilySize'] = df.SibSp + df.Parch + 1
    df['IsAlone'] = (df.FamilySize == 1).astype(int)
    df['HasCabin'] = df.Cabin.notna().astype(int)
    df['Deck'] = df.Cabin.str[0].fillna('U')
    df['TicketFreq'] = df.groupby('Ticket').Ticket.transform('count')
    df['FarePerPerson'] = df.Fare / df.TicketFreq
    return df

# A frequência do ticket usa treino + teste juntos (grupos que viajaram com o mesmo bilhete);
# não usa a variável alvo, então não há vazamento.
full = criar_atributos(pd.concat([train.drop(columns='Survived'), test], ignore_index=True))
# Idade: mediana por título, calculada só no treino
mediana_idade = full.iloc[:len(train)].groupby('Title').Age.median()
full['Age'] = full.Age.fillna(full.Title.map(mediana_idade))
X = full.iloc[:len(train)]
X_test = full.iloc[len(train):]
y = train.Survived
X[['Title', 'FamilySize', 'IsAlone', 'HasCabin', 'Deck', 'TicketFreq', 'FarePerPerson']].head()"""),
    code("""cat = ['Sex', 'Embarked', 'Title', 'Deck']
num = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin', 'TicketFreq', 'FarePerPerson']

prep = ColumnTransformer([
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat),
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]), num),
])"""),
    md("## 3. Comparação de modelos (validação cruzada estratificada, 5 folds)"),
    code("""modelos = {
    'Regressão Logística': LogisticRegression(max_iter=2000, C=0.5),
    'SVM (RBF)': SVC(C=1.0, gamma='scale'),
    'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=2, random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
resultados = {}
for nome, modelo in modelos.items():
    pipe = Pipeline([('prep', prep), ('model', modelo)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    resultados[nome] = scores
    print(f'{nome:22s} acurácia = {scores.mean():.4f} ± {scores.std():.4f}')"""),
    code("""melhor = max(resultados, key=lambda k: resultados[k].mean())
print('Melhor modelo na validação cruzada:', melhor)
pd.DataFrame(resultados).plot.box(figsize=(9, 3.5), title='Acurácia por fold', rot=10); plt.show()"""),
    md("## 4. Treino final e submissão"),
    code("""final = Pipeline([('prep', prep), ('model', modelos[melhor])]).fit(X, y)
sub = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final.predict(X_test).astype(int)})
sub.to_csv('submission.csv', index=False)
print(sub.Survived.value_counts(normalize=True).round(3))
sub.head()"""),
    md("""## Conclusões

- O título extraído do nome e o tamanho da família ajudam a recuperar informação de idade e de grupo de viagem.
- A validação cruzada estratificada evita escolher o modelo por sorte em uma única divisão treino/validação.
- Próximos passos: ajuste de hiperparâmetros com busca em grade e ensemble dos melhores modelos."""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "titanic.ipynb")
print("titanic.ipynb criado")
