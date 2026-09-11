import pandas as pd

# Create a simple DataFrame
data = {
    "Name": ["Amit", "Rahul", "Priya"],
    "Age": [20, 21, 20],
    "Marks": [85, 90, 78]
}

df = pd.DataFrame(data)

print("Student Data:")
print(df)

print("\nAverage Marks:")
print(df["Marks"].mean())