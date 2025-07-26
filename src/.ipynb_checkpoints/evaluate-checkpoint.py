from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load("app/model.joblib")

# Predict
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Save metrics in proper CSV format
metrics = {
    'accuracy': [acc]
}
df = pd.DataFrame(metrics)
df.to_csv("metrics.csv", index=False)
