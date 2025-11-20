from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

wine = load_wine()
x, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(x, y)

tree = DecisionTreeClassifier(max_depth=None)
tree.fit(X_train, y_train)

rules = export_text(tree, feature_names=wine.feature_names)
print("Reglas del Árbol de Decisión")
print(rules)

accuracy = tree.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy}")
