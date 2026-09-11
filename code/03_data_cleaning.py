import pandas as pd

# Load dataset
df = pd.read_csv("Telco-Customer-Churn_dataset.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Check result
print("Missing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)
import matplotlib.pyplot as plt

df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.savefig("Graphs/01_churn_distribution.png")
plt.show()
import matplotlib.pyplot as plt

df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.savefig("Graphs/01_churn_distribution.png")
plt.show()
# Churn rate by Contract Type

churn_rate = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

churn_rate.plot(kind="bar")

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

plt.savefig("Graphs/02_churn_by_contract.png")
plt.show()
# Churn rate by Contract Type

churn_rate = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

churn_rate.plot(kind="bar")

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

plt.savefig("Graphs/02_churn_by_contract.png")
plt.show()
# Churn Rate by Internet Service

churn_rate = df.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

churn_rate.plot(kind="bar")

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

plt.savefig("Graphs/03_churn_by_internet_service.png")
plt.show()
# Average Tenure by Churn

avg_tenure = df.groupby("Churn")["tenure"].mean()

avg_tenure.plot(kind="bar")

plt.title("Average Tenure by Churn")
plt.xlabel("Churn")
plt.ylabel("Average Tenure (Months)")

plt.xticks(rotation=0)

plt.savefig("Graphs/04_average_tenure_by_churn.png")
plt.show()
# Churn Rate by Payment Method

churn_rate = df.groupby("PaymentMethod")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

churn_rate.plot(kind="bar")

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig("Graphs/05_churn_by_payment_method.png")
plt.show()
# Churn Rate by Senior Citizen

churn_rate = df.groupby("SeniorCitizen")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

churn_rate.plot(kind="bar")

plt.title("Churn Rate by Senior Citizen")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

plt.savefig("Graphs/06_churn_by_senior_citizen.png")
plt.show()
# Churn Rate by Online Security

churn_rate = df.groupby("OnlineSecurity")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

churn_rate.plot(kind="bar")

plt.title("Churn Rate by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

plt.savefig("Graphs/07_churn_by_online_security.png")
plt.show()
# Churn Rate by Gender

churn_rate = df.groupby("gender")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

churn_rate.plot(kind="bar")

plt.title("Churn Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

plt.savefig("Graphs/08_churn_by_gender.png")
plt.show()
# Churn Rate by Phone Service

churn_rate = df.groupby("PhoneService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

churn_rate.plot(kind="bar")

plt.title("Churn Rate by Phone Service")
plt.xlabel("Phone Service")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

plt.savefig("Graphs/09_churn_by_phone_service.png")
plt.show()
# Save cleaned dataset

df.to_csv("Dataset/cleaned_telco_churn.csv", index=False)

print("Cleaned dataset saved successfully!")