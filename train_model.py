import pickle
import json

from sklearn.datasets       import fetch_openml
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# 1) load & split
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2) train
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

# 3) score
train_acc = clf.score(X_train, y_train)
test_acc  = clf.score(X_test, y_test)

# cross‐val on the full set
raw_cv_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
cv_mean = raw_cv_scores.mean().item()   # <-- .item() turns np.float64 → float

# 4) print nicely
print(f"Training accuracy: {train_acc:.3f}")
print(f"Test     accuracy: {test_acc:.3f}")
print(f"5-fold CV accuracies: {raw_cv_scores}")
print(f"Mean CV accuracy:    {cv_mean:.3f}")

# 5) save to JSON so your portfolio page can pick it up
results = {
    "train_acc": round(float(train_acc), 3),
    "test_acc":  round(float(test_acc),  3),
    "cv_mean":   round(cv_mean,          3)
}
with open("model_results.json", "w") as out:
    json.dump(results, out, indent=2)

# with open('mnist_model.pkl', 'wb') as f:
#     pickle.dump(clf, f)