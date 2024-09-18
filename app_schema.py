from pydantic import BaseModel

# Define the input structure
class CreditApplication(BaseModel):
    Account_type: int = 1
    Duration_of_Credit_month: int = 18
    Payment_Status_of_Previous_Credit: int = 4
    Purpose: int = 2
    Credit_Amount: int = 1049
    Savings_type: int = 1
    Length_of_current_employment: int = 2
    Instalment_per_cent: int = 4
    Marital_Status: int = 2
    Guarantors: int = 1
    Duration_in_Current_address: int = 4
    Most_valuable_available_asset: int = 2
    Age: int = 21
    Concurrent_Credits: int = 3
    Type_of_apartment: int = 1
    No_of_Credits_at_this_Bank: int = 1
    Occupation: int = 3
    No_of_dependents: int = 1
    Telephone: int = 1
    Foreign_Worker: int = 1
