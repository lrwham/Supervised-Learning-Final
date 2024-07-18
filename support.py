def get_scaled_data(data_df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    X = data_df.drop(columns=["Target"])
    y = data_df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def get_data():
    import pandas as pd

    data = pd.read_csv("data.csv", delimiter=";")
    # remove whitespace from column names
    data.columns = data.columns.str.strip()
    # convert 'Target' from 'Dropout', 'Enrolled', 'Graduated' to 0, 1, 2
    data["Target"] = data["Target"].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2})

    return data

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_preprocessed_data(data):
    cat_features = [
        "Marital status",
        "Application mode",
        "Application order",
        "Course",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Gender",
        "Tuition fees up to date",
        "Scholarship holder",
        "International",
        "Debtor",
        "Displaced",
        "Educational special needs",
        'Daytime/evening attendance',
    ]

    numeric_features = data.drop("Target", axis=1).columns.difference(cat_features)
    # print(f"Numeric features: {numeric_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), cat_features),  # One-hot encode the categorical columns
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),  # Standardize the remaining columns
        ]
    )

    data = get_data()

    X = data.drop(columns=["Target"])
    print(f"Shape of X before OneHot Encoding and Standardizing: {X.shape}")
    X = preprocessor.fit_transform(X)
    print(f"Shape of X after OneHot Encoding and Standardizing: {X.shape}")
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    return X_train, X_test, y_train, y_test, preprocessor
