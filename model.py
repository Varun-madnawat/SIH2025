import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

data = pd.read_csv("internship.csv", encoding='latin1')

data['stipend'] = data['stipend'].apply(lambda x: '0' if 'Unpaid' in str(x) else x)

data['stipend'] = data['stipend'].str.replace("/month", "").str.replace("lump sum", "").str.replace("₹", "")

data['Other Benifits'] = data['stipend'].str.split('+').apply(lambda x: x[-1].strip() if len(x) > 1 else None)
data['stipend'] = data['stipend'].str.split('+').str[0].str.strip()

data['min stipend'] = data['stipend'].str.split("-").str[0].str.extract(r'([\d,]+)', expand=False).str.replace(",", "")
data['max stipend'] = data['stipend'].str.split("-").str[-1].str.extract(r'([\d,]+)', expand=False).str.replace(",", "")

# Convert numeric columns to numeric type
numeric_columns = ['min stipend', 'max stipend']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

def weeks_to_months(duration):
    if 'Weeks' in duration:
        weeks = int(duration.split()[0])
        months = weeks / 4.33  # Average weeks in a month
        return f"{months:.2f} Months"
    else:
        return duration

data['duration'] = data['duration'].apply(weeks_to_months)
data['duration'] = data['duration'].str.replace(" Months", "").str.replace(" Month", "")
data["duration"] = data["duration"].apply(pd.to_numeric, errors='coerce')

Education = input("")
Skills = input("")
sector = input("")
location = input("")

# Filter internships based on user inputs
filtered_internships = data[
    (data['Sector'].str.contains(sector, case=False, na=False)) &
    (data['location'].str.contains(location, case=False, na=False))
].copy()

# Rank based on stipend (higher is better) and duration (longer is better) - you can adjust weighting as needed
filtered_internships['rank'] = filtered_internships['max stipend'].rank(ascending=False) + filtered_internships['duration'].rank(ascending=False)

# Sort by rank and get the top 3
best_internships = filtered_internships.sort_values(by='rank').head(3)
best_internships_without_rank = best_internships.drop(columns=["rank"])

# Convert to JSON format (list of dictionaries)
best_internships_json = best_internships_without_rank.to_dict(orient="records")


with open('output.json', 'w') as f:
    f.write(json.dumps(best_internships_json, indent=4))

# Print the formatted JSON
print(json.dumps(best_internships_json, indent=4))