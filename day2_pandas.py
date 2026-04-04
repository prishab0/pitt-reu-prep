import pandas as pd

#MARCH 16
# 1. We create a "Dictionary" - think of this as raw data
raw_data = {
    'cell_id': [1, 2, 3, 4, 5],
    'cell_type': ['Cancer', 'Immune', 'Cancer', 'Immune', 'Stroma'],
    'area_microns': [120, 85, 135, 90, 210],
    'is_nearby_tumor': [True, True, False, True, False],
    'protein_level': [0.8, 0.2, 0.9, 0.1, 0.5]  # New: 0.0 to 1.0 scale
}

# 2. We turn that raw data into a "DataFrame" (The Pandas Spreadsheet)
df = pd.DataFrame(raw_data)

# 3. Print the first few rows
print("--- THE WHOLE DATASET ---")
print(df.head())

# 4. Ask the dataset for "Info" (This tells you if data is missing or what type it is)
print("\n--- DATASET INFO ---")
print(df.info())

# 5. Get some quick math/statistics
print("\n--- SUMMARY STATISTICS ---")
print(df.describe()) 

# 6. Group by 'cell_type' and calculate the mean (average)
# This tells Pandas: "Bundle all Cancer together, all Immune together..."
averages = df.groupby('cell_type')['area_microns'].mean()

print("--- AVERAGE AREA PER CELL TYPE ---")
print(averages)

# 7. Count how many of each cell type we have
counts = df['cell_type'].value_counts()

print("\n--- CELL TYPE COUNTS ---")
print(counts)

#MARCH 17 - FILTERING DATA

# 1. Show only 'Immune' cells
# We are asking: "Where is the cell_type equal to Immune?"
immune_cells = df[df['cell_type'] == 'Immune']

print("--- ONLY IMMUNE CELLS ---")
print(immune_cells)

# 2. Show only LARGE cells (Area > 100)
# This is how we find "overgrown" cells
large_cells = df[df['area_microns'] > 100]

print("\n--- LARGE CELLS (>100 microns) ---")
print(large_cells)

# 3. Double Filter (The Pro Move)
# Find Cancer cells that are ALSO nearby a tumor
target_cells = df[(df['cell_type'] == 'Cancer') & (df['is_nearby_tumor'] == True)]

print("\n--- CANCER CELLS NEARBY TUMOR ---")
print(target_cells)

# --- MARCH 19: GROUPING DATA ---

# 1. Group by 'cell_type' and find the mean (average) area
# We are asking: "On average, how big is each type of cell?"
averages = df.groupby('cell_type')['area_microns'].mean()

print("\n--- AVERAGE AREA PER CELL TYPE ---")
print(averages)

# 2. Count how many of each cell type exist in the sample
# This is how we find the "Density" of the tumor
counts = df['cell_type'].value_counts()

print("\n--- CELL TYPE COUNTS ---")
print(counts)

# --- MARCH 20: DATA CLEANING ---

# 1. Check for missing values (The "Roll Call")
# This returns True for every empty cell, and we .sum() to count them
print("\n--- MISSING DATA COUNT ---")
print(df.isnull().sum())

# 2. Check for "Outliers"
# If a cell area is 0 or negative, something is wrong with the microscope
print("\n--- CHECKING FOR IMPOSSIBLE DATA ---")
glitch_check = df[df['area_microns'] <= 0]
print(f"Number of glitchy cells: {len(glitch_check)}")

#MARCH 21
# Now, let's use your MARCH 19 skills to find the protein average!
protein_summary = df.groupby('cell_type')['protein_level'].mean()
print("\n--- AVERAGE PROTEIN LEVEL BY CELL TYPE ---")
print(protein_summary)


# --- MARCH 22: THRESHOLDING (THE ON/OFF SWITCH) ---

# Define our threshold: Anything above 0.5 is "High Expression"
# This creates a new column based on a rule
df['is_high_expression'] = df['protein_level'] > 0.5

print("\n--- NEW COLUMN: HIGH EXPRESSION FLAG ---")
print(df[['cell_id', 'cell_type', 'protein_level', 'is_high_expression']])

# Count how many "High Expression" cells we have
high_count = df['is_high_expression'].value_counts()
print("\n--- COUNT OF HIGH VS LOW EXPRESSION ---")
print(high_count)


