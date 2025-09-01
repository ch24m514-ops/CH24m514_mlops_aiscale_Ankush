# src/test_api.py
import requests
import time
from rich.console import Console
from rich.table import Table

URL = "http://127.0.0.1:8000/predict"
console = Console()

# Different passenger scenarios
test_cases = [
    {
        "name": "First-class female (likely survivor)",
        "data": {
            "Pclass": 1,
            "Sex": "female",
            "Age": 38.0,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 71.2833,
            "Embarked": "C"
        }
    },
    {
        "name": "Third-class young male",
        "data": {
            "Pclass": 3,
            "Sex": "male",
            "Age": 19.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 7.8958,
            "Embarked": "S"
        }
    },
    {
        "name": "Second-class child",
        "data": {
            "Pclass": 2,
            "Sex": "male",
            "Age": 4.0,
            "SibSp": 1,
            "Parch": 1,
            "Fare": 23.0,
            "Embarked": "S"
        }
    },
    {
        "name": "Third-class female (solo)",
        "data": {
            "Pclass": 3,
            "Sex": "female",
            "Age": 30.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 7.75,
            "Embarked": "Q"
        }
    },
    {
        "name": "Wealthy first-class male",
        "data": {
            "Pclass": 1,
            "Sex": "male",
            "Age": 45.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 83.475,
            "Embarked": "S"
        }
    }
]

def main():
    table = Table(title="🚢 Titanic Survival API — Test Report")
    table.add_column("Case", style="cyan", justify="left")
    table.add_column("Input Summary", style="magenta")
    table.add_column("Prediction", style="green")
    table.add_column("Status", style="bold")
    table.add_column("Time (ms)", justify="right")

    for case in test_cases:
        start = time.time()
        try:
            resp = requests.post(URL, json=case["data"])
            elapsed = round((time.time() - start) * 1000, 2)
            resp.raise_for_status()
            prediction = resp.json()

            summary = f"{case['data']['Sex']}, {case['data']['Age']} yrs, Class {case['data']['Pclass']}"
            table.add_row(case["name"], summary, str(prediction), "✅", str(elapsed))

        except Exception as e:
            elapsed = round((time.time() - start) * 1000, 2)
            table.add_row(case["name"], "-", f"Error: {e}", "❌", str(elapsed))

    console.print(table)

if __name__ == "__main__":
    main()
