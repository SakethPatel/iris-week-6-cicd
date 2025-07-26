from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = joblib.load("app/model.joblib")
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

with open("metrics.csv", "w") as f:
    f.write(f"accuracy: {acc:.4f}")
