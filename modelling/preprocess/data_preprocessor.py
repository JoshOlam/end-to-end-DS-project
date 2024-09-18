import os
import json

import pandas as pd

from sklearn.preprocessing import StandardScaler

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Set random state for reproducibility
RANDOM_STATE =int(os.getenv("RANDOM_STATE", ""))

# Specify the path to the data file
DATA_PATH =os.getenv("DATA_PATH", "")

PREPROCESSED_DATA_PATH =os.getenv("PREPROCESSED_DATA_PATH", "")
CONFIG_COLUMNS = os.getenv("CONFIG_COLUMNS", "")
CONFIG_CAT_COL = os.getenv("CONFIG_CAT_COL", "")
CONFIG_SCALED_COL = os.getenv("CONFIG_SCALED_COL", "")

class Preprocessor:
    def __init__(self, data_path: str) -> None:
        assert isinstance(data_path, str), f"data_path {data_path!r} must be a string and not {type(data_path)!r}"
        assert data_path.endswith(".csv"), f"data_path {data_path!r} must be a csv file"
        self.data_path = data_path
        self.scaler = StandardScaler()
    
    def read_data(self)->pd.DataFrame:
        try:
            return pd.read_csv(self.data_path)
        except:
            return None
    
    def preprocess(self, data: pd.DataFrame, index: str="user_id", target: str="label")->pd.DataFrame:
        # Ensure the received data is of expected class
        assert isinstance(data, pd.DataFrame), f"input dataframe must be of instance pd.Dataframe, not {type(data)!r}"
        print("Received data is a Pandas DataFrame")

        # Ensure user_id is present in the data
        assert index in data.columns and target in data.columns, f"Both {index!r} and {target!r} must be present in dataframe"
        print(f"Both {index!r} and {target!r} are present in the DataFrame")

        print("Engineering Features...")

        # Create a copy of the data
        df = data.copy(deep=True)
        print("\t...Insurance copy created")

        # Write the config for columns
        with open(CONFIG_COLUMNS, "w") as file:
            json.dump({"columns": [col for col in df.columns]}, file)

        # Set the index as the index of the dataframe
        df.set_index(index, inplace=True)
        print(f"\t...Index set to {index!r}")

        # Replace whitespace(" ") with underscore("_")
        df.columns = df.columns.str.replace(" ", "_")
        df.columns = df.columns.str.replace("(", "")
        df.columns = df.columns.str.replace(")", "")
        df.columns = df.columns.str.lower()
        print("\t...Whitespaces replaced with underscores and columns converted to lower strings")

        # Split into target and features
        X = df.drop(columns=target)
        y = df[target]

        # Handle categorical features
        # Get the columns with less than 10 unique values
        cat = X.columns[X.nunique() <= 10]
        # Convert the data type to categorical
        X_cat = X[cat].astype("category")
        # Write the categorical columns to config file
        with open(CONFIG_CAT_COL, "w") as file:
            json.dump({col: list(X_cat[col].unique()) for col in X_cat.columns}, file)
        X_cat = pd.get_dummies(data=X_cat, drop_first=True, dtype=int)
        X = pd.concat(objs=[X_cat, X.drop(columns=cat)], axis=1)
        print("\t...Categorical features handled")
        print("Feature engineering completed")

        X_scaled = self.scaler.fit_transform(X)
        with open(CONFIG_SCALED_COL, "w") as file:
            json.dump({"columns": [col for col in X.columns]}, file)

        # convert the scaled predictor values into a dataframe
        pd.DataFrame(X_scaled, columns=X.columns).to_csv(PREPROCESSED_DATA_PATH, index=False)

        return X_scaled, y
    
    def predict_preprocess(self, input_data: dict):
        assert isinstance(input_data, dict), f"Input data must be a dict, not {type(input_data)!r}"
    
        # Load saved configurations for columns and categorical mappings
        with open(CONFIG_COLUMNS, "r") as file:
            self.config_columns = json.load(file)["columns"]
        with open(CONFIG_CAT_COL, "r") as file:
            self.cat_columns = json.load(file)
        with open(CONFIG_SCALED_COL, "r") as file:
            self.scaled_columns = json.load(file)["columns"]

        # Expected columns for preprocessing
        expected_columns = self.config_columns[2:]
        expected_columns = [str(col).replace(" ", "_").replace("(", "").replace(")", "").lower() for col in expected_columns]

        # Create a DataFrame from the input data
        df = pd.DataFrame([input_data])
        df.columns = df.columns.str.replace(" ", "_").str.replace("(", "").str.replace(")", "").str.lower()

        # Check for expected columns
        for column in expected_columns:
            if column not in df.columns:
                print("Col: ", column)
                df[column] = 0  # Add missing columns with default value 0

        # Initialize a DataFrame to hold one-hot encoded features
        encoded_data = pd.DataFrame(index=df.index)

        # Process categorical columns
        cat_columns = [col for col in df.columns if col in self.cat_columns]
        for col in cat_columns:
            accepted_values = self.cat_columns[col]
            input_value = df[col].values[0]

            # Validate the input value is within the acceptable range
            if input_value not in accepted_values:
                raise ValueError(f"Value {input_value} in column {col} is not within the expected range {min(accepted_values)}-{max(accepted_values)}")

            # Create one-hot encoded columns
            for val in accepted_values:
                col_name = f"{col}_{val}"
                encoded_data[col_name] = [1 if input_value == val else 0]

        # Process non-categorical columns
        non_cat_columns = [col for col in df.columns if col not in self.cat_columns]
        for col in non_cat_columns:
            encoded_data[col] = df[col]

        # Ensure all expected columns are present
        for col in expected_columns:
            if col not in encoded_data.columns:
                encoded_data[col] = 0  # Add missing columns with default value 0

        # Reorder columns to match the training feature set
        encoded_data = encoded_data[self.scaled_columns]

        # Step 3: Scaling the numerical features
        try:
            X_scaled = self.scaler.transform(encoded_data)
        except Exception as e:
            data = self.read_data()
            self.preprocess(data=data)
            X_scaled = self.scaler.transform(encoded_data)

        return X_scaled


if __name__ == "__main__":
    pro = Preprocessor(data_path=DATA_PATH)
    data = pro.read_data()
    prep = pro.preprocess(data=data)
