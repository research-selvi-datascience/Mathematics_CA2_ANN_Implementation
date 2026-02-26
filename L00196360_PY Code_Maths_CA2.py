# Databricks notebook source
# MAGIC %md
# MAGIC # Paddy Dataset
# MAGIC ### Introduction:
# MAGIC Agriculture occupies nearly one-third of the Earth’s land surface and plays a critical role in global food security. Rice, cultivated from paddy seeds, is a staple food for almost half of the world’s population. With the continuous growth of the global population and increasing pressure on agricultural resources, improving rice productivity has become an essential challenge. Recent advances in machine learning have enabled data-driven approaches for yield prediction and decision support in agriculture. Among these, Artificial Neural Networks (ANNs) have shown strong potential in modeling complex, nonlinear relationships between crop yield and influencing factors such as climate, soil properties, and farm management practices. This study aims to enhance rice production by developing an ANN-based model to predict paddy yield. The ANN algorithm is implemented from scratch to provide better control over the training process and to address challenges such as overfitting and computational efficiency.  The dataset is available at: https://archive.ics.uci.edu/dataset/1186/paddy+dataset  
# MAGIC
# MAGIC ### Business Understanding
# MAGIC Accurate prediction of paddy yield is essential for farmers, agricultural planners, and policymakers to improve decision-making and optimize resource utilization. This project focuses on predicting paddy yield using a wide range of input features, including climatic factors (temperature, rainfall, wind speed, wind direction, and relative humidity), Soil characteristics and land type, Cultivation area and agronomic practices and Nutrient and fertilizer application. By leveraging these features, the model aims to provide reliable yield predictions that can support improved crop planning and sustainable agricultural practices. 
# MAGIC
# MAGIC ### Research Hypothesis
# MAGIC An Artificial Neural Network can predict paddy yield with higher accuracy while maintaining computational efficiency by effectively capturing nonlinear relationships among climatic, soil, and agronomic factors.
# MAGIC
# MAGIC ### Research Question
# MAGIC Can an Artificial Neural Network accurately predict paddy yield using climatic, soil, and management-related features?
# MAGIC
# MAGIC #### Importing the Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import math

# COMMAND ----------

# MAGIC %md
# MAGIC #### Importing the dataset

# COMMAND ----------

url = "https://raw.githubusercontent.com/research-selvi-datascience/Mathematics_CA2_ANN_Implementation/main/paddydataset.csv"
paddy_data = pd.read_csv(url)

paddy_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preprocessing
# MAGIC
# MAGIC #### Datatypes:
# MAGIC The datatypes of the variables are mixed - integer, continous and categorical. The dataset has 2,789 instances and 45 features, that are essential in predicting the paddy yield.  The dataset has variables relating to soil, climate, land, area (or county), and much more which are essential for predicting the yield.

# COMMAND ----------

paddy_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checking missing values and inconsistencies

# COMMAND ----------

paddy_data.isnull().sum()

# COMMAND ----------

duplicated_rows = paddy_data[paddy_data.duplicated()]
duplicated_rows

# COMMAND ----------

# MAGIC %md
# MAGIC Although several records appear identical across climatic and agronomic variables, they represent distinct field-level observations from different locations and cultivation settings. These repetitions arise due to shared regional weather conditions and standardized farming practices and are therefore retained in the dataset.

# COMMAND ----------

# Finding the value counts of the categorical variables:

categorical_columns = paddy_data.select_dtypes(include=['object'])

for column in categorical_columns:
    print(paddy_data[column].value_counts())
       		

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In Agriblock,  6 individual unique categories are present, which are the name of the counties.
# MAGIC In variety, 3 different unique species of rice are present.
# MAGIC In soil types: 2 classes are present
# MAGIC Nursery, 2 classes are present
# MAGIC All wind direction features, 6 subcategories are presnt.
# MAGIC
# MAGIC All categorical attributes, including agriblock, rice variety, soil type, nursery type, and wind direction, were identified as nominal variables with no inherent ordering. These variables were therefore encoded using one-hot encoding to ensure compatibility with the Artificial Neural Network model.
# MAGIC
# MAGIC Wind direction variables exhibit circular characteristics. To preserve their cyclical nature and avoid artificial ordering, sine-cosine transformation was applied by mapping directional categories to angular values and encoding them as sine and cosine components.
# MAGIC Wind direction categories (N, NE, SSE, etc.) are directions on a compass, and each compass direction corresponds to a fixed angle in degrees. So not inventing any new values, but using standard meteorological compass angles.
# MAGIC The Category for the Angle (degrees): N	0°, 
# MAGIC NE	45°, 
# MAGIC ENE	67.5°, 
# MAGIC E	90°, 
# MAGIC ESE	112.5°, 
# MAGIC SE	135°, 
# MAGIC SSE	157.5°, 
# MAGIC S	180°, 
# MAGIC SSW	202.5° , 
# MAGIC SW	225°,  
# MAGIC WSW	247.5°,  
# MAGIC W	270°,  
# MAGIC WNW	292.5°,  
# MAGIC NW	315°,  
# MAGIC NNW	337.5°.     This mapping comes from standard compass direction definitions. 
# MAGIC
# MAGIC Each angle is converted using sine & cosine
# MAGIC
# MAGIC For each angle θ:
# MAGIC
# MAGIC x=sin(θ)
# MAGIC y=cos(θ)
# MAGIC
# MAGIC This creates two numerical features per wind direction column.    So, in this manner, Wind direction categories were transformed into angular representations and encoded using sine and cosine functions to preserve their cyclical nature.
# MAGIC

# COMMAND ----------

import numpy as np

# Step 1: Define standard compass angles
wind_angle_map = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}

# Step 2: List wind direction columns
wind_columns = [
    "Wind Direction_D1_D30",
    "Wind Direction_D31_D60",
    "Wind Direction_D61_D90",
    "Wind Direction_D91_D120"
]

# Step 3: Apply sine-cosine encoding
for col in wind_columns:
    angles = paddy_data[col].map(wind_angle_map)

    # Safety check for unmapped values
    if angles.isnull().any():
        missing = paddy_data[col][angles.isnull()].unique()
        raise ValueError(f"Unmapped wind direction values in {col}: {missing}")

    paddy_data[col + "_sin"] = np.sin(np.deg2rad(angles))
    paddy_data[col + "_cos"] = np.cos(np.deg2rad(angles))

# Step 4: Drop original categorical wind columns
paddy_data.drop(columns=wind_columns, inplace=True)


# COMMAND ----------

paddy_data.head()

# COMMAND ----------

# One- hOt encoding the other variables
variables = ['Agriblock', 'Variety', 'Soil Types', 'Nursery']

for col in paddy_data[variables]:
    paddy_data = pd.concat([paddy_data.drop(col, axis=1), pd.get_dummies(paddy_data[col], prefix=col, drop_first = True).astype(int)], axis=1)

paddy_data.head()


# COMMAND ----------

# Checking for null values after encoding

paddy_data.isnull().sum()

# COMMAND ----------

# Checking the datatypes
paddy_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Descriptive Statistics

# COMMAND ----------

paddy_data.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC In all the variables, mean values are lower than the median showing the existence of high lower values.
# MAGIC
# MAGIC ### Histogram of yield

# COMMAND ----------

plt.figure(figsize=(10, 6))
sns.histplot(x='Paddy yield(in Kg)', data=paddy_data, kde=True)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC THe yield shows multimodal distribution, with multiple peaks in the distribution. This indicates that there are multiple groups of data points with different characteristics that are contributing to the overall distribution. This could be due to various factors such as differences in soil quality, weather conditions, or farming practices among different regions or fields. There a high proportion of data points with yield around 30000. 
# MAGIC
# MAGIC ### Correlation

# COMMAND ----------

plt.figure(figsize=(14,14))
correlation = paddy_data.corr()

correlation_df = pd.DataFrame(correlation, columns=paddy_data.columns)
correlation_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multicollinearity check: Variance inflation factor

# COMMAND ----------

# Severe Multicollinearity is being observed
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Separate features (X) and target (y)
X = paddy_data.drop(columns=['Paddy yield(in Kg)'])

# Step 2: Ensure X contains only numeric columns
X = X.select_dtypes(include=['int64', 'float64'])

# Step 3: Compute VIF correctly
vif = pd.DataFrame()
vif['feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif.sort_values(by='VIF', ascending=False)


# COMMAND ----------

# MAGIC %md
# MAGIC Variance Inflation Factor analysis indicated severe multicollinearity due to deterministic relationships among area-based inputs, rainfall metrics, and cyclical encodings. Since Artificial Neural Networks do not assume linear independence among features, redundant variables were removed based on domain knowledge rather than VIF thresholds.
# MAGIC
# MAGIC During exploratory data analysis, certain variables were found to be exact duplicates of others, indicating data-entry or column-alignment issues. Specifically, LP_nurseryarea(in Tonnes) was identical to Hectares, and Weed28D_thiobencarb matched Nursery area (Cents) across all observations. These variables were therefore removed to maintain data integrity and avoid spurious learning.
# MAGIC
# MAGIC Seedrate: 25 * Hectares,   
# MAGIC LP_Mainfield:  Hectares * 25/2  ,   
# MAGIC Nursery area: Hectares * 20 ,  
# MAGIC LP_Nursery area:  Hecatres ,  
# MAGIC DAP_20:  Hectares * 40,    
# MAGIC Weed:   Hectares  * 2,   
# MAGIC Urea:  27.13 * Hectares,   
# MAGIC Potash:   Hectares * 10 (app.)  , 
# MAGIC Micronutriesnts: Hectares * 15,              
# MAGIC Pest_60Days:   Hectares *  600
# MAGIC
# MAGIC So, these variables are of:
# MAGIC Input i=ci * Hectares
# MAGIC
# MAGIC Management inputs such as seed rate, fertilizer dosage, pesticide application, and nursery parameters were found to follow fixed agronomic recommendations proportional to cultivated area. These input variables are all derived from the feature Hecatres thus representing the deterministic Linear function of the Hectare variable. According to the ML, these are not actual features but they are all constants, scaled by the land area. Mathematically thay add zero degrees of freedom. As these variables exhibited no independent variability, they were excluded from the model to prevent deterministic redundancy and shortcut learning in the ANN.
# MAGIC #### Retaining the Hectares feature, and dropping the linear deterministic features
# MAGIC

# COMMAND ----------

paddy_data = paddy_data.drop(
    columns=['Seedrate(in Kg)',	'LP_Mainfield(in Tonnes)',	'Nursery area (Cents)',	'LP_nurseryarea(in Tonnes)',	'DAP_20days',	'Weed28D_thiobencarb',	'Urea_40Days',	'Potassh_50Days',	'Micronutrients_70Days',	'Pest_60Day(in ml)']
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Multicollinearity is again checked after dropping the high VIF features

# COMMAND ----------

# Severe Multicollinearity is being observed
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Separate features (X) and target (y)
X = paddy_data.drop(columns=['Paddy yield(in Kg)'])

# Step 2: Ensure X contains only numeric columns
X = X.select_dtypes(include=['int64', 'float64'])

# Step 3: Compute VIF correctly
vif = pd.DataFrame()
vif['feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif.sort_values(by='VIF', ascending=False)


# COMMAND ----------

# MAGIC %md
# MAGIC Although several predictors exhibited high VIF values due to categorical encoding and cyclic transformations, these were retained as neural networks are not constrained by multicollinearity in the same manner as linear regression models. Only exact linear dependencies were removed.

# COMMAND ----------

# MAGIC %md
# MAGIC Seedrate (in Kg): Seedrate refers to the quantity of paddy seed sown for cultivation, measured in kilograms. The Agricultural role determines plant density, affects competition for nutrients, water, and sunlight, excess seedrate means overcrowding and low seedrate means poor crop stand. Seedrate is directly proportional to land area (hectares): Seedrate = Recommended rate per hectare * Area.       Hence, it does not add independent information once area is known.
# MAGIC
# MAGIC 2. LP_Mainfield (in Tonnes):  Represents the quantity of inputs used during land preparation in the main field, such as organic matter, Soil amendments, Machinery or labor effort (converted to tonnes equivalent). Its Agricultural role is improves soil structure, enhances root penetration, supports early crop establishment. LP_Mainfield scales linearly with cultivated area: LP_Mainfield * Hectares.      Thus, it duplicates area information.
# MAGIC
# MAGIC 3. Nursery area (Cents):         Area allocated for raising paddy seedlings before transplanting, measured in cents
# MAGIC (1 cent = 1/100 acre). Its Agricultural role is seedling production, controls transplanting density, affects early plant vigor. Nursery area is a fixed fraction of total cultivated area:     Nursery area≈10–15%×Hectares. Therefore, it is deterministically derived from hectares.
# MAGIC
# MAGIC 4. LP_nurseryarea (in Tonnes):  Represents land preparation effort or inputs used specifically in the nursery area, measured in tonnes.    It's Agricultural role eEnsures healthy seedling growth, improves seedling survival after transplanting. LP_nurseryarea depends on: LP_nurseryarea * Nursery area * Hectares. Thus, it adds no new independent signal.
# MAGIC
# MAGIC 5. 30DAI (in mm): (30 Days After Irrigation).  Represents artificial irrigation water supplied (in mm) during the first 30 days of crop growth. Agricultural role: Maintains standing water, supports seedling establishment  and compensates rainfall deficit. Irrigation (AI) is complementary to rainfall: Rainfall + Irrigation = Total water requirement. Hence: High rainfall THEN low AI and Low rainfall THEN high AI. This creates perfect negative correlation.
# MAGIC
# MAGIC 6. 30_50DAI (in mm):   Artificial irrigation applied between 30-50 days after planting (tillering stage). It's Agricultural role encourages tiller formation, supports vegetative growth. Hence, redundant with rainfall variables.
# MAGIC
# MAGIC 7. 51_70AI (in mm) : Artificial irrigation during panicle initiation stage. It's Agricultural role is critical stage for yield determination. Water stress here drastically reduces yield. Deterministically linked with rainfall in the same stage.
# MAGIC
# MAGIC 8.  71_105DAI (in mm): Irrigation supplied during grain filling and maturity stages. iT'S Agricultural role maintains grain weight, prevents spikelet sterility.  Complements rainfall; therefore redundant if rainfall is already included.
# MAGIC
# MAGIC Seedrate, LP_Mainfield, Nursery area	ARE all Direct function of land area. 
# MAGIC LP_nurseryarea is derived from nursery area. 
# MAGIC AI variables	are perfect complements of rainfall. 
# MAGIC
# MAGIC Several agronomic inputs such as seed rate, land preparation quantities, nursery area,  were found to be deterministically derived from cultivated area. To avoid redundant information and improve computational efficiency, these variables were excluded from the neural network model while retaining area as primary explanatory factors.

# COMMAND ----------

paddy_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Checking outliers:

# COMMAND ----------

# Checking outliers:
plt.figure(figsize=(20,30))
for i, col in enumerate(paddy_data.columns):
    plt.subplot(11,4,i+1)   # 7 rows, 4 columns
    sns.boxplot(x=paddy_data[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Testing for normality using Shapiro and Q-Q plots
# MAGIC

# COMMAND ----------

from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot

for col in paddy_data.columns:
    stat, p = shapiro(paddy_data[col])
    print(f'Shapiro test for {col}:')
    print(f'Statistic = {stat}, p-value = {p}')
    print('Normal distribution' if p > 0.05 else 'Not normal distribution')
    print('')


# COMMAND ----------

# MAGIC %md
# MAGIC Shapiro-Wilk tests indicated that none of the predictor variables followed a normal distribution. However, normality is not a prerequisite for Artificial Neural Networks. Since ANN models are distribution-free and rely on numerical stability rather than parametric assumptions, the features were standardized using z-score normalization (StandardScaler) prior to training.
# MAGIC
# MAGIC #### Renaming the column names for ML Compatability

# COMMAND ----------

# Renaming the feature names:
paddy_data.rename(columns={'Hectares ': 'Hectares', '30DRain( in mm)': '30DRain_mm', '30DAI(in mm)': '30DAI_mm', '30_50DRain( in mm)': '30_50DRain_mm', '30_50DAI(in mm)': '30_50DAI_mm', '51_70DRain(in mm)': '51_70DRain_mm', '51_70AI(in mm)': '51_70AI_mm', '71_105DRain(in mm)': '71_105DRain_mm', '71_105DAI(in mm)': '71_105DAI_mm', 'Min temp_D1_D30': 'Min_temp_D1_D30', 'Max temp_D1_D30': 'Max_temp_D1_D30', 'Min temp_D31_D60': 'Min_temp_D31_D60', 'Max temp_D31_D60': 'Max_temp_D31_D60', 'Min temp_D61_D90': 'Min_temp_D61_D90', 'Max temp_D61_D90': 'Max_temp_D61_D90', 'Min temp_D91_D120': 'Min_temp_D91_D120', 'Max temp_D91_D120': 'Max_temp_D91_D120', 'Inst Wind Speed_D1_D30(in Knots)': 'Inst_Wind_Speed_D1_D30', 'Inst Wind Speed_D31_D60(in Knots)': 'Inst_Wind_Speed_D31_D60', 'Inst Wind Speed_D61_D90(in Knots)': 'Inst_Wind_Speed_D61_D90', 'Inst Wind Speed_D91_D120(in Knots)': 'Inst_Wind_Speed_D91_D120', 'Relative Humidity_D1_D30': 'Relative_Humidity_D1_D30', 'Relative Humidity_D31_D60': 'Relative_Humidity_D31_D60', 'Relative Humidity_D61_D90': 'Relative_Humidity_D61_D90', 'Relative Humidity_D91_D120': 'Relative_Humidity_D91_D120', 'Trash(in bundles)': 'Trash_bundles', 'Paddy yield(in Kg)': 'Paddy_yield_kg', 'Wind Direction_D1_D30_sin': 'Wind_Direction_D1_D30_sin', 'Wind Direction_D1_D30_cos': 'Wind_Direction_D1_D30_cos', 'Wind Direction_D31_D60_sin': 'Wind_Direction_D31_D60_sin', 'Wind Direction_D31_D60_cos': 'Wind_Direction_D31_D60_cos', 'Wind Direction_D61_D90_sin': 'Wind_Direction_D61_D90_sin', 'Wind_Direction_D61_D90_cos': 'Wind_Direction_D61_D90_cos', 'Wind Direction_D91_D120_sin': 'Wind_Direction_D91_D120_sin', 'Wind Direction_D91_D120_cos': 'Wind_Direction_D91_D120_cos', 'Agriblock_Cuddalore': 'Agriblock_Cuddalore', 'Agriblock_Krishnagiri': 'Agriblock_Krishnagiri', 'Agriblock_Kumbakonam': 'Agriblock_Kumbakonam', 'Agriblock_Panruti': 'Agriblock_Panruti', 'Agriblock_Sankarapuram': 'Agriblock_Sankarapuram', 'Variety_delux ponni': 'Variety_delux_ponni', 'Variety_ponmani': 'Variety_ponmani', 'Soil Types_clay': 'Soil_Types_clay', 'Nursery_wet': 'Nursery_wet'}, inplace=True)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking the datatypes of features before Modelling

# COMMAND ----------

paddy_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining X and y for standardising and Modelling:

# COMMAND ----------

X = paddy_data.drop(columns=['Paddy_yield_kg'])
y = paddy_data['Paddy_yield_kg']

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Type that are continuous numeric, are standardised as it improve gradient descent & convergence. Sin/Cos (angle encoding) are not encoded as it has already normalized to [-1,1]. One-hot categorical features are not encoded, as 0/1 values carry categorical meaning.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Splitting the data in 80:20 using train_test_split

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnosing the X_train

# COMMAND ----------

X_train.info()

# COMMAND ----------

X_train.iloc[:, 0:26]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standardising the X_train, X_test, y_train and y_test. 
# MAGIC Converting the y to numpy and reshaping to tune with the Neural Network standards. The continuous numeric features are standardised manually (X - np.mean(X))/np.std(X). The One hot categorical features and the Sin/Cos are finally concatenated with the standardised X features.

# COMMAND ----------


# Select continuous columns to standardize (0–25)
cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scaling of the numerical features:
# MAGIC 1. Continuous features standardized only : Standardization (X - mean)/std ensures zero mean and unit variance, which helps neural networks converge faster.
# MAGIC
# MAGIC 2. Sin/cos and one-hot categorical features are untouched
# MAGIC
# MAGIC 3. Categorical one-hot encodings remain in their original scale (0/1 for one-hot, -1 to 1 for sin/cos).
# MAGIC
# MAGIC 4. Target scaling: Standardizing y_train and y_test is standard when using tanh or sigmoid outputs, and reshaping ensures it matches neural network expected shape (n_samples, 1).
# MAGIC
# MAGIC 5. Final concatenation : Combines standardized and unchanged features back into a single DataFrame, ready for NN input.
# MAGIC
# MAGIC ### Checking the descriptive statistics of the X_train to find the range

# COMMAND ----------

X_train.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Finding the range of y_train_scaled for deciding on the activation functions.

# COMMAND ----------

y_min = np.min(y_train_scaled)
y_max = np.max(y_train_scaled)
y_mean = np.mean(y_train_scaled)

print(f"Min yield: {y_min}")
print(f"Max yield: {y_max}")
print(f"Mean yield: {y_mean:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forward Pass:
# MAGIC
# MAGIC #### X ------ Z(1) ------- a(1) ------- Z(2) ------ Y PRED --------L
# MAGIC 1. Z(1) = W(1).X                            
# MAGIC 2. a(1) = f(Z(1))                                     
# MAGIC 3. Z(2) = W(2).a(1)
# MAGIC 4. Y_PRED = f(Z(2))
# MAGIC 5. L = (Y-Y_PRED)**2
# MAGIC
# MAGIC ##### Initializing ANN parameters
# MAGIC # Experiment 1:    Architecture:
# MAGIC 1. Inputs: 43 features (X)
# MAGIC 2. Hidden layer: 1 neuron
# MAGIC 3. Output layer: 1 neuron
# MAGIC 4. No biases
# MAGIC 5. Activation function: Sigmoid for hidden and RELU for output
# MAGIC #### Checking the shape of the X_train and X_test

# COMMAND ----------

X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standardising of X_train, X_test, y_train, y_test and reshaping.

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting the parameters and initialising the weights

# COMMAND ----------

# Setting the random seed:
np.random.seed(42)

# Inputs (43 features)  and 1 hidden neuron and 1 output (Yield)
n_inputs = X_train.shape[1]
n_hidden = 1
n_output = 1
learning_rate = 0.01

# Randomly initialize weights
# W1 = np.random.randn(n_hidden, n_inputs) *   # weights input -> hidden (1x43)
# W2 = np.random.randn(n_output, n_hidden)  # weights hidden -> output (1x1)

W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1. / n_hidden)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining the activation function and its derivative. 
# MAGIC ### Forward Pass: Finding Z1

# COMMAND ----------

# Activation function

# tanh activation for hidden layer
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# --- FORWARD PASS ---

# Step 1: Hidden layer pre-activation
Z1 = np.dot(X_train, W1.T)
print(Z1)
print(f"min of Z1 is {np.min(Z1)} and max of Z1 is  {np.max(Z1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC The hidden layer pre-activation values Z1 were observed to lie approximately within the range [−1.44,1.42]. Applying different activation functions transforms these values differently: Sigmoid squashes them into the range [0,1],  tanh maps them into [−1,1], ReLU sets negative values to zero while keeping positive values unchanged. In the first five baseline experiments, these activation functions were evaluated under identical conditions to compare their impact on training convergence and generalisation performance.
# MAGIC ### Step 2: Hidden layer activation

# COMMAND ----------

A1 = tanh(Z1) 
print(A1)
print(f"min of A1 is {np.min(A1)} and max of A1 is  {np.max(A1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC With tanh activation function, the range of A1 is [- 0.89, 0.89]. 
# MAGIC ### Step 3: Output layer pre-activation

# COMMAND ----------

Z2 = np.dot(A1, W2) 
print(Z2)
print(f"min of Z2 is {np.min(Z2)} and max of Z2 is  {np.max(Z2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Output layer activation (prediction)
# MAGIC The scaled target variable ranges approximately from -1.85 to 1.77. Sigmoid restricts outputs to [0,1], ReLU restricts outputs to [0,∞), and tanh restricts outputs to [-1,1]. Since the target exceeds these bounded ranges, using these activations in the output layer would constrain predictions and introduce systematic error. Therefore, a linear activation is most appropriate for regression tasks.
# MAGIC Hidden layer activations can be nonlinear (tanh, ReLU, sigmoid). But output layer activation depends on: 
# MAGIC Classification (sigmoid/softmax), Regression (linear). Range of Linear function: (−∞,+∞). No restriction.
# MAGIC Can represent the full scaled target range. That’s why linear is standard for regression.

# COMMAND ----------

y_train_pred = Z2    
print(y_train_pred)
print(f"min of y_train_pred is {np.min(y_train_pred)} and max of y_train_pred is  {np.max(y_train_pred)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculating the loss : MSE & RMSE

# COMMAND ----------

print("Z1 (hidden pre-activation):\n", Z1)
print("A1 (hidden activation):\n", A1)
print("Z2 (output pre-activation):\n", Z2)
print("Y_pred (output prediction):\n", y_train_pred)

loss = np.mean((y_train_scaled - y_train_pred)**2)
rmse = np.sqrt(loss)  
print(f" MSE Loss: {loss} and  RMSE Loss: {rmse} on training data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next Step:  Backpropogation   
# MAGIC 1. Gradient of loss w.r.t output         
# MAGIC Loss w.r.t prediction:   L = (y-y_pred)**2 

# COMMAND ----------

dL_dy_pred = 2 * (y_train_pred - y_train_scaled) / len(y_train_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.  Since output is linear: DY_PRED_DZ2 = (Z2): 

# COMMAND ----------

dy_pred_dZ2 = 1

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Gradient of Z2 w.r.t W2

# COMMAND ----------

dZ2_dW2 = A1

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Chain Rule: 
# MAGIC DL_D_YPRED * D_YPRED_DZ2 * DZ2_DW2

# COMMAND ----------

dL_dZ2 = dL_dy_pred * dy_pred_dZ2
dL_dW2 = dL_dZ2.T @ dZ2_dW2

# COMMAND ----------

# MAGIC %md
# MAGIC 5.  Finding W1:
# MAGIC DL/D_YPRED * D_YPRED/DZ2 * DZ2/DA1 * DA1/DZ1 * DZ1 / DW1

# COMMAND ----------

dZ2_dA1 = W2

# COMMAND ----------

# MAGIC %md
# MAGIC 6. Gradient of Loss w.r.t A1

# COMMAND ----------

dL_dA1 = dL_dZ2 @ dZ2_dA1

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Gradient of A1 w.r.t Z1 (tanh derivative)

# COMMAND ----------

dA1_dZ1 = 1 - np.tanh(Z1)**2

# COMMAND ----------

# MAGIC %md
# MAGIC 8. Gradient of Loss w.r.t Z1

# COMMAND ----------

dL_dZ1 = dL_dA1 * dA1_dZ1

# COMMAND ----------

# MAGIC %md
# MAGIC 9. Gradient of Z1 w.r.t W1
# MAGIC 10. Gradient of Loss w.r.t W1

# COMMAND ----------

dL_dW1 = dL_dZ1.T @ X_train

# COMMAND ----------

# MAGIC %md
# MAGIC 11. WEIGHT UPDATE

# COMMAND ----------

lr = 0.01
W2 = W2 - (lr * dL_dW2)
W1 = W1 - (lr * dL_dW1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forward pass (again with the updated weights)

# COMMAND ----------

Z1 = X_train @ W1.T
A1 = tanh(Z1)

Z2 = A1 @ W2.T
y_train_pred = Z2   # linear output

# COMMAND ----------

# MAGIC %md
# MAGIC ### Computing the Loss again:

# COMMAND ----------

loss = np.mean((y_train_scaled - y_train_pred)**2)
rmse = np.sqrt(loss)  
print(f" MSE Loss: {loss} and  RMSE Loss: {rmse}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### TESTING ON THE X_TEST

# COMMAND ----------

Z1_test = np.dot(X_test, W1.T)
A1_test = tanh(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = Z2_test

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test MSE Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Converting the scaled y values back to its original units for calculating the MSE, RMSE and relative % of RMSE to its mean.

# COMMAND ----------

y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion for Experiment 1:
# MAGIC Architecture: 43 input features, 1 hidden layer with 1 tanh neuron, 1 linear output neuron. A linear activation was used in the output layer because the task is regression with continuous targets. 
# MAGIC
# MAGIC Backpropagation Verification:
# MAGIC Initial loss: MSE = 1.0581; RMSE = 1.0287. After one gradient update:  MSE = 1.0540;  RMSE = 1.0266.  The reduction in loss confirms that the forward pass is correctly implemented, Gradients are correctly computed and backpropagation updates weights in the correct direction.  Even with only one hidden neuron, saturated tanh activation and a single gradient descent step, the model still shows measurable improvement. This validates the correctness of the implementation. 
# MAGIC
# MAGIC The tanh function: for large positive inputs, it becomes very close to +1 and for large negative inputs, it becomes very close to −1. Around 0, it changes smoothly and strongly. So tanh has a steep middle region and a flat regions near +1 and −1. If the Z1 values are large (like 3, 4, -5, etc.), then tanh(Z1)≈ +1 or −1. So, the neuron output is stuck near +1 or −1. It’s not changing much anymore, saturating.
# MAGIC In the middle (near 0), the derivative is large. But near ±1, derivative is almost 0. Then weight updates become very tiny. So learning becomes very slow, vanishing gradient. Z1 ∈ [−1.44,1.42]. Those values are already moderately large. So many tanh outputs are close to ±1. So A1 ≈ ±1. Derivative small, weight updates small so loss decreases slowly. 
# MAGIC
# MAGIC The improvement is small as the limited reduction in loss is expected due to very low model capacity (only one hidden neuron), activation saturation (tanh operating in flat region) and vanishing gradients and a single optimisation step. Experiment 1 serves as a baseline model and a check for backpropagation and a demonstration of how activation saturation limits learning. The small but consistent decrease in loss confirms correct gradient derivation and weight updates, while highlighting the limitations of insufficient model capacity and activation saturation.
# MAGIC
# MAGIC # Experiment 2: 
# MAGIC ### 43 inputs, 1 Sigmoid hidden neuron, 1 Linear output (yield)

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# =========================
# EXPERIMENT 2
# Sigmoid hidden neuron
# Regression output (paddy yield)
# =========================

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)

# -------------------------
# Network architecture
# -------------------------
n_inputs = X_train.shape[1]   # 43 features
n_hidden = 1                  # single hidden neuron
n_output = 1                  # regression output

# -------------------------
# Weight initialization
# -------------------------
W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1. / n_hidden)

# -------------------------
# Activation functions
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    # derivative w.r.t Z, using activation output
    return a * (1 - a)

# -------------------------
# Forward pass
# -------------------------
Z1 = X_train @ W1.T            # (N x 1)
A1 = sigmoid(Z1)               # (N x 1)

Z2 = A1 @ W2.T                 # (N x 1)
y_train_pred = Z2              # linear output

# -------------------------
# Loss (Mean Squared Error)
# -------------------------
loss = np.mean((y_train_scaled - y_train_pred) ** 2)

print("Initial forward pass")
print("Z1 (hidden pre-activation):\n", Z1[:5])
print("A1 (hidden activation):\n", A1[:5])
print("Predictions:\n", y_train_pred[:5])
rmse = np.sqrt(loss)  
print(f" MSE Loss: {loss} and RMSE Loss: {rmse}")


# COMMAND ----------

# -------------------------
# Backpropagation
# -------------------------
# dL/dŷ
dL_dy_pred = 2 * (y_train_pred - y_train_scaled) / len(y_train_scaled)

# Output layer gradients
dL_dZ2 = dL_dy_pred                  # linear output
dL_dW2 = dL_dZ2.T @ A1               # (1 x 1)

# Hidden layer gradients
dL_dA1 = dL_dZ2 @ W2                 # (N x 1)
dA1_dZ1 = sigmoid_derivative(A1)     # (N x 1)
dL_dZ1 = dL_dA1 * dA1_dZ1             # (N x 1)

# Gradient w.r.t W1
dL_dW1 = dL_dZ1.T @ X_train           # (1 x 43)

# -------------------------
# Gradient descent update
# -------------------------
learning_rate = 0.01

W2 -= learning_rate * dL_dW2
W1 -= learning_rate * dL_dW1

# -------------------------
# Forward pass after update
# -------------------------
Z1 = X_train @ W1.T
A1 = sigmoid(Z1)

Z2 = A1 @ W2.T
y_train_pred = Z2

loss = np.mean((y_train_scaled - y_train_pred) ** 2)

print("\nAfter one gradient descent step")
rmse = np.sqrt(loss)  
print(f"updated MSE Loss: {loss} and RMSE Loss: {rmse} on the trained data")

# COMMAND ----------

# TESTING ON THE X_TEST

Z1_test = np.dot(X_test, W1.T)
A1_test = sigmoid(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = Z2_test

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# COMMAND ----------

# Scaling back to the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion for experiment 2:  
# MAGIC
# MAGIC Architecture: 43 input features, 1 hidden neuron with sigmoid activation, 1 linear output neuron
# MAGIC
# MAGIC The hidden pre-activation values Z1 contained both large positive and negative magnitudes due to random weight initialization and input features containing negative values (e.g., sine/cosine wind direction encoding). As a result, many inputs to the sigmoid activation lay in its extreme regions leading to Sigmoid saturation. The sigmoid activation function outputs values in the range: (0, 1). When Z1 is very positive, sigmoid ≈ 1. When Z1 is very negative, sigmoid ≈ 0. In both cases, the derivative becomes very small. This phenomenon is called activation saturation. Because the derivative is close to zero, The backpropagated gradients become very small and the Weight updates are extremely small. Learning proceeds very slowly, leading to vanishing gradients.
# MAGIC
# MAGIC The loss Behaviour:    
# MAGIC Initial loss: MSE = 1.03436; RMSE = 1.01704. After one update: MSE = 1.03363; RMSE = 1.01668. The small reduction in loss confirms that the backpropagation is implemented correctly, Gradients are flowing but learning is slow due to sigmoid saturation
# MAGIC
# MAGIC Experiment 2 demonstrates that Sigmoid activation is prone to saturation when inputs have large magnitude. 
# MAGIC Saturation leads to very small gradients. Small gradients result in slow learning. With only one hidden neuron, the model capacity is also limited. Thus, this experiment serves as a validation of backpropagation correctness and a demonstration of vanishing gradient effects with sigmoid
# MAGIC
# MAGIC # Experiment 3:
# MAGIC #### Network Architecture: 43 inputs, ReLU hidden neuron (hidden layer) and Linear output layer (paddy yield)

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------


# =========================
# EXPERIMENT 3
# ReLU hidden neuron
# Linear output (paddy yield)
# =========================

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)

# Network architecture
n_inputs = X_train.shape[1]   # 43 features
n_hidden = 1                  # single hidden neuron
n_output = 1                  # regression output

# Weight initialization
W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs) # (1 x 43)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1./n_hidden)  # (1 x 1)

# -------------------------
# ReLU activation
# -------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(z):
    return (z > 0).astype(float)

# -------------------------
# Forward pass
# -------------------------
Z1 = X_train @ W1.T           # Hidden pre-activation
A1 = relu(Z1)                 # Hidden activation

Z2 = A1 @ W2.T                # Linear output
y_train_pred = Z2             # Regression prediction

# Compute initial loss (MSE)
loss = np.mean((y_train_scaled - y_train_pred)**2)

print("Initial forward pass")
print("Z1 (hidden pre-activation):\n", Z1[:5])
print("A1 (hidden activation):\n", A1[:5])
print("Predictions:\n", y_train_pred[:5])
rmse = np.sqrt(loss)  
print(f" MSE Loss: {loss} and RMSE Loss: {rmse}")

# COMMAND ----------

# -------------------------
# Backpropagation
# -------------------------

# dL/dŷ
dL_dy_pred = 2 * (y_train_pred - y_train_scaled) / len(y_train_scaled)

# Output layer gradients
dL_dZ2 = dL_dy_pred                 # linear output
dL_dW2 = dL_dZ2.T @ A1              # (1 x 1)

# Hidden layer gradients
dL_dA1 = dL_dZ2 @ W2                # (N x 1)
dA1_dZ1 = relu_derivative(Z1)       # (N x 1)
dL_dZ1 = dL_dA1 * dA1_dZ1           # (N x 1)

# Gradient w.r.t W1
dL_dW1 = dL_dZ1.T @ X_train         # (1 x 43)

# -------------------------
# Gradient descent update
# -------------------------
learning_rate = 0.01

W2 -= learning_rate * dL_dW2
W1 -= learning_rate * dL_dW1

# -------------------------
# Forward pass after update
# -------------------------
Z1 = X_train @ W1.T
A1 = relu(Z1)

Z2 = A1 @ W2.T
y_train_pred = Z2

loss = np.mean((y_train_scaled - y_train_pred)**2)

print("\nAfter one gradient descent step")
rmse = np.sqrt(loss)  
print(f" updated MSE Loss: {loss} and  RMSE Loss: {rmse}")

# COMMAND ----------

# TESTING ON THE X_TEST

Z1_test = np.dot(X_test, W1.T)
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = Z2_test

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")


# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion for Experiment 3
# MAGIC Hidden Layer Behaviour: ReLU activation applies negative pre-activations output to 0. Positive pre-activations output are linear (unchanged). So no saturation for positive values, neuron can represent magnitude (not compressed to [-1,1] or [0,1]), stronger gradient flow when active. Some values become 0 (inactive neuron) while others retain positive magnitude
# MAGIC
# MAGIC Loss Behaviour: Initial loss: MSE = 1.05414; RMSE = 1.02671. After one update: MSE = 1.05056; RMSE = 1.02497. The loss reduction is larger than in the sigmoid experiment.
# MAGIC
# MAGIC Sigmoid problem is, it saturates at 0 or 1, and the derivative becomes very small.  But with 
# MAGIC ReLU, for positive inputs, derivative will be larger, gradients do not shrink enabling faster learning. This reduces vanishing gradient effects for active neurons.
# MAGIC
# MAGIC Experiment 3 demonstrates that ReLU activation improves gradient flow compared to sigmoid. Since the derivative of ReLU is 1 for positive inputs, gradients remain strong and learning proceeds faster. The larger reduction in loss after one update confirms improved optimisation dynamics. However, with only one hidden neuron, model capacity remains limited.

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 4:
# MAGIC Network Setup: 43 Input features, Linear hidden, Linear 1 output:  (2 layer network)
# MAGIC
# MAGIC Training Loss:  MSE Loss: 1.0980132630459383 and RMSE Loss: 1.0478612804402776
# MAGIC
# MAGIC After one gradient descent step:   updated MSE Loss: 1.0868631196243292 and  RMSE Loss: 1.0425272752423935
# MAGIC
# MAGIC Test Loss: 1.0525
# MAGIC Test RMSE Loss: 1.0259185935943271;  Test MSE (original units): 89952967.44 KG ;  Test RMSE (original units): 9484.35 KG;  Relative RMSE (% of mean yield): 41.76%

# COMMAND ----------

# =========================
# EXPERIMENT 4
# Linear hidden neuron
# linear output (paddy yield)
# =========================

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)

# -------------------------
# Network architecture
# -------------------------
n_inputs = X_train.shape[1]   # 43 features
n_hidden = 1                  # single hidden neuron
n_output = 1                  # regression output

# -------------------------
# Weight initialization
# -------------------------
W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1. / n_hidden)

# -------------------------
# Activation functions
# -------------------------
def linear(x):
    return x

def linear_derivative(x):
   return np.ones_like(x)

# -------------------------
# Forward pass
# -------------------------
Z1 = X_train @ W1.T            # (N x 1)
A1 = linear(Z1)               # (N x 1)

Z2 = A1 @ W2.T                 # (N x 1)
y_train_pred = linear(Z2)              # linear output

# -------------------------
# Loss (Mean Squared Error)
# -------------------------
loss = np.mean((y_train_scaled - y_train_pred) ** 2)

print("Initial forward pass")
print("Z1 (hidden pre-activation):\n", Z1[:5])
print("A1 (hidden activation):\n", A1[:5])
print("Predictions:\n", y_train_pred[:5])
rmse = np.sqrt(loss)  
print(f" MSE Loss: {loss} and RMSE Loss: {rmse}")
print(f" Predictions min: {np.min(y_train_pred)} and Predictions max: {np.max(y_train_pred)}")


# COMMAND ----------

# -------------------------
# Backpropagation
# -------------------------

# dL/dŷ
dL_dy_pred = 2 * (y_train_pred - y_train_scaled) / len(y_train_scaled)

# Output layer gradients
dL_dZ2 = dL_dy_pred * linear_derivative(Z2)  # (N x 1)
dL_dW2 = dL_dZ2.T @ A1              # (1 x 1)                 

# Hidden layer gradients
dL_dA1 = dL_dZ2 @ W2                # (N x 1)
dA1_dZ1 = linear_derivative(Z1)       # (N x 1)
dL_dZ1 = dL_dA1 * dA1_dZ1           # (N x 1)

# Gradient w.r.t W1
dL_dW1 = dL_dZ1.T @ X_train         # (1 x 43)

# -------------------------
# Gradient descent update
# -------------------------
learning_rate = 0.01

W2 -= learning_rate * dL_dW2
W1 -= learning_rate * dL_dW1

# -------------------------
# Forward pass after update
# -------------------------
Z1 = X_train @ W1.T
A1 = linear(Z1)

Z2 = A1 @ W2.T
y_train_pred = linear(Z2)

loss = np.mean((y_train_scaled - y_train_pred)**2)

print("\nAfter one gradient descent step")
rmse = np.sqrt(loss)  
print(f" updated MSE Loss: {loss} and  RMSE Loss: {rmse}")

# COMMAND ----------

# TESTING ON THE X_TEST

Z1_test = np.dot(X_test, W1.T)
A1_test = linear(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = linear(Z2_test)

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC Linear hidden, Linear output, Predictions range: -0.43 to +0.43. Target is scaled (likely near -1 to +1). 
# MAGIC Because the linear output predictions already lie inside (-1, 1), the model naturally produces values within the range of tanh. The output is not exploding, so it is reasonable to test tanh in the output layer, since tanh outputs in (-1, 1). 
# MAGIC
# MAGIC # Experiment 5:
# MAGIC Network setup: 43 Input features, 1 linear hidden layer, 1 tanh output layer:
# MAGIC
# MAGIC Training results:
# MAGIC MSE Loss: 1.091544175428133 and RMSE Loss: 1.0447699150665342
# MAGIC
# MAGIC Forward pass after Updated weights: After one gradient descent step;  updated MSE Loss: 1.0822600403591056 and  RMSE Loss: 1.0403172786987178
# MAGIC
# MAGIC Test error:
# MAGIC Test Loss: 1.0535, Test RMSE Loss: 1.026413737938091, Test MSE (original units): 90039817.31 KG, Test RMSE (original units): 9488.93 KG, Relative RMSE (% of mean yield): 41.65%

# COMMAND ----------

# =========================
# EXPERIMENT 4
# Linear hidden neuron
# tanh output (paddy yield)
# =========================

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)

# -------------------------
# Network architecture
# -------------------------
n_inputs = X_train.shape[1]   # 43 features
n_hidden = 1                  # single hidden neuron
n_output = 1                 

# -------------------------
# Weight initialization
# -------------------------
W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1. / n_hidden)

# -------------------------
# Activation functions
# -------------------------
def linear(x):
    return x

def linear_derivative(x):
   return np.ones_like(x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# -------------------------
# Forward pass
# -------------------------
Z1 = X_train @ W1.T            # (N x 1)
A1 = linear(Z1)               # (N x 1)

Z2 = A1 @ W2.T                 # (N x 1)
y_train_pred = tanh(Z2)              # linear output

# -------------------------
# Loss (Mean Squared Error)
# -------------------------
loss = np.mean((y_train_scaled - y_train_pred) ** 2)

print("Initial forward pass")
print("Z1 (hidden pre-activation):\n", Z1[:5])
print("A1 (hidden activation):\n", A1[:5])
print("Predictions:\n", y_train_pred[:5])
rmse = np.sqrt(loss)  
print(f" MSE Loss: {loss} and RMSE Loss: {rmse}")
print(f" Predictions min: {np.min(y_train_pred)} and Predictions max: {np.max(y_train_pred)}")


# COMMAND ----------

# -------------------------
# Backpropagation
# -------------------------

# dL/dŷ
dL_dy_pred = 2 * (y_train_pred - y_train_scaled) / len(y_train_scaled)

# Output layer gradients
dL_dZ2 = dL_dy_pred * tanh_derivative(Z2)  # (N x 1)
dL_dW2 = dL_dZ2.T @ A1              # (1 x 1)                 

# Hidden layer gradients
dL_dA1 = dL_dZ2 @ W2                # (N x 1)
dA1_dZ1 = linear_derivative(Z1)       # (N x 1)
dL_dZ1 = dL_dA1 * dA1_dZ1           # (N x 1)

# Gradient w.r.t W1
dL_dW1 = dL_dZ1.T @ X_train         # (1 x 43)

# -------------------------
# Gradient descent update
# -------------------------
learning_rate = 0.01

W2 -= learning_rate * dL_dW2
W1 -= learning_rate * dL_dW1

# -------------------------
# Forward pass after update
# -------------------------
Z1 = X_train @ W1.T
A1 = linear(Z1)

Z2 = A1 @ W2.T
y_train_pred = tanh(Z2)

loss = np.mean((y_train_scaled - y_train_pred)**2)

print("\nAfter one gradient descent step")
rmse = np.sqrt(loss)  
print(f" updated MSE Loss: {loss} and  RMSE Loss: {rmse}")

# COMMAND ----------

# TESTING ON THE X_TEST

Z1_test = np.dot(X_test, W1.T)
A1_test = linear(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = tanh(Z2_test)

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 6: (Using Epocs)
# MAGIC Architecture: 43 input features, 1 hidden neuron, tanh activation (hidden layer), 1 linear output, (regression), Learning rate = 0.01.

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# MAGIC %md
# MAGIC Architecture:
# MAGIC 43 features, 1, 1. Learning rate=0.01, epochs=1000, tanh for the hidden layer ivation function and linear for the output layer activation.

# COMMAND ----------

# Setup:

np.random.seed(42)

n_inputs = X_train.shape[1]
n_hidden = 1
n_output = 1
lr = 0.01
epochs = 1000
loss_history = []

# Initialize weights
W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1./n_hidden)


# COMMAND ----------

# Activation Functions
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# COMMAND ----------

# Epoch Loop (Forward and backward)

for epoch in range(epochs):

    # ---------- FORWARD PASS ----------

    # Z1 = XW1ᵀ
    Z1 = np.dot(X_train, W1.T)          # (N, 1)

    # A1 = tanh(Z1)
    A1 = tanh(Z1)                    # (N, 1)

    # Z2 = A1W2ᵀ
    Z2 = np.dot(A1, W2.T)               # (N, 1)

    # Output (linear)
    y_pred = Z2                         # (N, 1)

    # ---------- LOSS ----------
    loss = np.mean((y_train_scaled - y_pred) ** 2)
    loss_history.append(loss)

    # ---------- BACKPROPAGATION ----------

    # dL/dy_pred
    dL_dy = -2 * (y_train_scaled - y_pred) / len(y_train_scaled)   # (N, 1)

    # dy_pred/dZ2 = 1 (linear)
    dL_dZ2 = dL_dy                         # (N, 1)

    # dZ2/dW2 = A1
    dL_dW2 = np.dot(dL_dZ2.T, A1)          # (1, 1)

    # dZ2/dA1 = W2
    dL_dA1 = np.dot(dL_dZ2, W2)            # (N, 1)

    # dA1/dZ1 (N, 1)
    dL_dZ1 = dL_dA1 * tanh_derivative(A1)

    # dZ1/dW1 = X
    dL_dW1 = np.dot(dL_dZ1.T, X_train)     # (1, 43)

    # ---------- UPDATE WEIGHTS ----------

    W2 -= lr * dL_dW2
    W1 -= lr * dL_dW1

    # ---------- LOG ----------
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

plt.figure(figsize=(10,6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Gradient Descent Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

# TESTING ON THE X_TEST
Z1_test = np.dot(X_test, W1.T)
A1_test = tanh(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = Z2_test

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")


# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
print(f"Mean y_test mean yield is {mean_yield} and std is {np.std(y_test)}" )
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion:
# MAGIC Architecture: 43 input features, 1 hidden neuron, tanh activation (hidden layer), 1 linear output, (regression), Learning rate = 0.01. Trained for multiple epochs using gradient descent.  In earlier experiments, only one forward and backward pass was employed for verifying gradient correctness. It show effect of activation functions on gradient size. In this experiment, orward + backward pass repeated over many epochs and the Weights updated gradually while the Loss tracked over time. This is actual training, not just a gradient check.  
# MAGIC
# MAGIC Convergence Behaviour:
# MAGIC Epoch 0, the Loss = 1.0581. Rapid decrease during early epochs with the minimum training loss ≈ 0.0746 around epoch 300. But slight increase afterward showing mild overfitting. This shows that Gradient descent is working correctly and the network is learning meaningful structure. Training dynamics can only be observed over multiple epochs. 
# MAGIC
# MAGIC Test Performance:
# MAGIC Test RMSE = 2957 kg; Mean yield = 22,712 kg and the Relative RMSE = 13.02% of mean yield. The model captures meaningful patterns in the data, as the prediction error is much smaller than the natural variability (std ≈ 9006 kg). However, accuracy is still moderate, indicating room for improvement.
# MAGIC
# MAGIC The experiment shows strong convergence initially, confirms correct implementation and the slow improvement later shows the limited model capacity (only 1 hidden neuron). tanh saturation reduces gradient magnitude. With a slight overfitting after ~300 epochs. To observe this behaviour, multiple epochs are necessary. This also shows the backpropagation correctness (single step). While the model converges and captures structure, its limited capacity and tanh saturation restrict further performance gains.
# MAGIC
# MAGIC Unlike previous experiments that verified gradient correctness with a single update, this experiment performs full training over many epochs. The loss decreases steadily from 1.0581 to around 0.0746, confirming proper convergence. However, learning slows due to the use of a single hidden neuron and tanh saturation. The final test RMSE of 2957 kg, or 13% of the mean yield, indicates reasonable but limited predictive performance.
# MAGIC
# MAGIC # Experiment 7: Linear Hidden Layer (No Activation)
# MAGIC Architecture: 43 input features, 1 hidden neuron, Linear activation (hidden layer), 1 linear output. Gradient Descent over multiple epochs. This effectively makes the entire network a linear transformation of the input features as Linear and a Linear = Linear model.  So mathematically, this behaves like linear regression (just written as a 2-layer network).

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# Setup:

np.random.seed(42)

n_inputs = X_train.shape[1]
n_hidden = 1
n_output = 1
lr = 0.01
epochs = 1000
loss_history = []

# Initialize weights
W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1./n_hidden)

# Activation Functions:

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)



# COMMAND ----------

# Epoch Loop (Forward and backward)

for epoch in range(epochs):

    # ---------- FORWARD PASS ----------

    # Z1 = XW1ᵀ
    Z1 = np.dot(X_train, W1.T)          # (N, 1)

    # A1 = linear(Z1)
    A1 = linear(Z1)                    # (N, 1)

    # Z2 = A1W2ᵀ
    Z2 = np.dot(A1, W2.T)               # (N, 1)

    # Output (linear)
    y_pred = Z2                         # (N, 1)

    # ---------- LOSS ----------
    loss = np.mean((y_train_scaled - y_pred) ** 2)
    loss_history.append(loss)

    # ---------- BACKPROPAGATION ----------

    # dL/dy_pred
    dL_dy = -2 * (y_train_scaled - y_pred) / len(y_train_scaled)   # (N, 1)

    # dy_pred/dZ2 = 1 (linear)
    dL_dZ2 = dL_dy                         # (N, 1)

    # dZ2/dW2 = A1
    dL_dW2 = np.dot(dL_dZ2.T, A1)          # (1, 1)

    # dZ2/dA1 = W2
    dL_dA1 = np.dot(dL_dZ2, W2)            # (N, 1)

    # dA1/dZ1 (N, 1)
    dL_dZ1 = dL_dA1 * linear_derivative(A1)

    # dZ1/dW1 = X
    dL_dW1 = np.dot(dL_dZ1.T, X_train)     # (1, 43)

    # ---------- UPDATE WEIGHTS ----------

    W2 -= lr * dL_dW2
    W1 -= lr * dL_dW1

    # ---------- LOG ----------
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

plt.figure(figsize=(10,6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Gradient Descent Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------

# TESTING ON THE X_TEST
Z1_test = np.dot(X_test, W1.T)
A1_test = linear(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = Z2_test

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
print(f"Mean y_test mean yield is {mean_yield} and std is {np.std(y_test)}" )
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC Training Convergence: 
# MAGIC Epoch 0   : 1.0980
# MAGIC Epoch 100 : 0.1272
# MAGIC Epoch 200 : 0.0183
# MAGIC Epoch 900 : 0.0138
# MAGIC
# MAGIC THis shows extremely fast convergence, and smooth monotonic decrease and no instability. This looks like no overfitting spike and a very stable optimization
# MAGIC
# MAGIC Test Performance: 
# MAGIC Test MSE = 0.0140; Test RMSE = 0.1183; Original Units: Test RMSE = 1093.81 kg abd Relative RMSE = 4.82% of mean yield. This is better than Experiment 4 in terms of the RMSE relative error of 4.82%
# MAGIC Linear work better because the problem appears mostly linear and tanh introduced saturation resulting in vanishing gradients. With only 1 hidden neuron, adding nonlinearity did not increase expressive power. 
# MAGIC This is a classical linear regression. The dataset is largely linearly predictable. This is a powerful result in comparison with the previous experiments.
# MAGIC
# MAGIC Using a linear hidden layer significantly improved both convergence speed and predictive accuracy. The model achieved a test RMSE of 1093 kg (4.82% of the mean yield), substantially outperforming the tanh-based model. This indicates that the relationship between inputs and yield is largely linear, and that introducing nonlinearity with limited capacity can degrade performance due to saturation effects.

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 8:  Increasing hidden neurons (5) to capture more patterns.
# MAGIC Network setup: Hidden neurons: 5;  Activation: linear in hidden layer, tanh output;  Weights initialized with Xavier/variance-scaled method: W1 ~ N(0, 1/n_inputs), W2 ~ N(0, 1/n_hidden);   Learning rate: 0.01;  
# MAGIC Epochs: 1000;  Loss: Mean Squared Error (MSE)

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# -----------------------------
# 2. Network configuration
# -----------------------------
np.random.seed(42)

n_inputs = X_train.shape[1]
n_hidden = 5     # 5 hidden neurons
n_output = 1

# Randomly initialize weights
W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1./n_hidden)

# Learning rate
lr = 0.01

# -----------------------------
# 3. Activation function
# -----------------------------
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# -----------------------------

# 4. Training loop
# -----------------------------
loss_history = []
epochs = 1000

for epoch in range(epochs):
    # --- Forward pass ---
    Z1 = np.dot(X_train, W1.T)       # (n_samples, n_hidden)
    A1 = Z1                 # (n_samples, n_hidden)
    Z2 = np.dot(A1, W2.T)            # (n_samples, n_output)
    y_pred = tanh(Z2)                      

    # --- Loss ---
    loss = np.mean((y_train_scaled - y_pred)**2)
    loss_history.append(loss)

# Step 1: dL/dy_pred
    dL_dy = 2 * (y_pred - y_train_scaled) / y_train_scaled.shape[0]

# Step 2: apply tanh derivative (output layer)
    dL_dZ2 = dL_dy * tanh_derivative(Z2)

# Step 3: gradient w.r.t W2
    dL_dW2 = np.dot(dL_dZ2.T, A1)

# Step 4: propagate to hidden
    dL_dA1 = np.dot(dL_dZ2, W2)

# Hidden layer is linear → derivative = 1
    dL_dZ1 = dL_dA1   # no multiplication needed

# Step 5: gradient w.r.t W1
    dL_dW1 = np.dot(dL_dZ1.T, X_train)

    # --- Update weights ---
    W1 -= lr * dL_dW1
    W2 -= lr * dL_dW2

    # --- Print loss every 100 epochs ---
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# COMMAND ----------

plt.figure(figsize=(10,6))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Gradient Descent Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


# COMMAND ----------

# TESTING ON THE X_TEST
Z1_test = np.dot(X_test, W1.T)
A1_test = Z1_test
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = tanh(Z2_test)

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
print(f"Mean y_test mean yield is {mean_yield} and std is {np.std(y_test)}" )
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 8: Linear (1 Hidden Layer) and tanh (Output Layer)
# MAGIC In comparison to the previous experiments, the model yielded a low RMSE relative score of 12.20%
# MAGIC
# MAGIC # Experiment 9: Tanh (1 Hidden Layer) and Linear (Output layer)
# MAGIC Network setup: Hidden neurons: 5; Activation: tanh in hidden layer, linear output; Weights initialized with Xavier/variance-scaled method: W1 ~ N(0, 1/n_inputs), W2 ~ N(0, 1/n_hidden); Learning rate: 0.01;
# MAGIC Epochs: 1000; Loss: Mean Squared Error (MSE)
# MAGIC
# MAGIC Training: Forward and backward passes repeated over 1000 epochs. Loss decreases steadily, confirming proper gradient flow and backpropagation. 
# MAGIC
# MAGIC Testing results (X_test): Test MSE (scaled): 0.0136; Test RMSE (scaled): 0.1165; Test RMSE in original units: 1077.01 kg; Relative RMSE: 4.74% of mean yield.
# MAGIC
# MAGIC The model shows smooth convergence, with loss dropping rapidly in early epochs and slowly stabilizing. 
# MAGIC Using 5 neurons allows the network to better capture the complexity of the input features, reducing both training and test error compared to single-neuron models. Tanh activation Works well for hidden neurons; the activations avoid extreme saturation due to moderate weight initialization. Linear output layer is suitable for regression, allowing both positive and negative predictions. Relative RMSE of 4.74% indicates strong predictive capability given the natural variability of yield (~9000 kg standard deviation).
# MAGIC
# MAGIC Conclusion:
# MAGIC
# MAGIC The network uses 43 inputs, 5 hidden neurons with tanh activation, and a single linear output for regression. Weights were initialized with Xavier scaling, and the model was trained for 1000 epochs using a learning rate of 0.01. Training loss decreased smoothly from 1.0668 at epoch 0 to 0.0149 by epoch 900, demonstrating proper gradient flow and stable backpropagation. On the test set, the MSE was 0.0136 (scaled) with an RMSE of 1077 kg in original units, corresponding to 4.74% of the mean yield. Increasing the hidden layer size from one to five neurons allowed the network to better capture input complexity, while tanh activations provided stable hidden representations. The linear output layer ensured predictions could take both positive and negative values. Overall, the experiment confirms that the network learns effectively over epochs, shows strong convergence, and generalizes well to unseen data.

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 10:
# MAGIC In this experiment, systematically vary the number of hidden neurons in a single hidden layer: 10 neurons, 
# MAGIC 20 neurons, 30 neurons. All other factors are kept constant: same dataset, same learning rate (0.01), same hidden layer activation (tanh) and linear output layer, same number of epochs (1000). This isolates the effect of model capacity.

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------


# Network configuration

np.random.seed(42)

n_inputs = X_train.shape[1]
n_hidden_list = [10, 20, 30]  
n_output = 1
lr = 0.01
epochs = 1000

# Activation function
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# -----------------------------
# 2. Training loop for different hidden neurons
# -----------------------------
for n_hidden in n_hidden_list:
    # Initialize weights
    W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
    W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1. / n_hidden)

    loss_history = []

    for epoch in range(epochs):
        # --- Forward pass ---
        Z1 = np.dot(X_train, W1.T)
        A1 = tanh(Z1)
        Z2 = np.dot(A1, W2.T)
        y_pred = Z2  # linear output

        # --- Loss ---
        loss = np.mean((y_train_scaled - y_pred)**2)
        loss_history.append(loss)

        # --- Backpropagation ---
        dL_dy = 2 * (y_pred - y_train_scaled) / y_train_scaled.shape[0]
        dL_dW2 = np.dot(dL_dy.T, A1)
        dL_dA1 = np.dot(dL_dy, W2)
        dL_dZ1 = dL_dA1 * tanh_derivative(Z1)
        dL_dW1 = np.dot(dL_dZ1.T, X_train)

        # --- Update weights ---
        W1 -= lr * dL_dW1
        W2 -= lr * dL_dW2

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Hidden neurons: {n_hidden}, Epoch {epoch}, Loss: {loss:.4f}")

    # --- Plot training loss ---
    plt.plot(loss_history, label=f'{n_hidden} neurons')

plt.title("Gradient Descent Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Final training losses (≈ epoch 900)
# MAGIC Hidden neurons	Final loss
# MAGIC 10	0.6448
# MAGIC 20	0.8195
# MAGIC 30	0.7000
# MAGIC
# MAGIC All three significantly outperform earlier experiments (loss ≈ 1.0 - 1.9).This shows increasing hidden neurons dramatically reduces underfitting. 10 neurons performing best here. Although more neurons increase capacity, 
# MAGIC they also make optimisation harder, gradients interfere with each other and learning rate may be suboptimal. 
# MAGIC With 10 neurons, capacity is sufficient and optimisation is easy. 20 - 30 neurons, capacity increases, but learning becomes less efficient. This is a classic bias - variance trade-off in action. Initial loss differences (Epoch 0): 10 neurons starts at loss ≈ 9.14, 20 neurons starts at ≈ 5.69, 30 neurons starts at ≈ 9.88. This is due to random weight initialisation, larger hidden layers producing larger initial activations. Curves flatten after ~700 epochs, convergence. This confirms correct gradient computation,  stable learning rate and correct backpropagation.  Paddy yield depends on multiple interacting climate factors. A single neuron cannot represent this. Multiple neurons allow the network to learn different nonlinear responses and different climate regimes.

# COMMAND ----------

# -----------------------------
# 3. Test set evaluation
# -----------------------------
# TESTING ON THE X_TEST
Z1_test = np.dot(X_test, W1.T)
A1_test = tanh(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = Z2_test

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
print(f"Mean y_test mean yield is {mean_yield} and std is {np.std(y_test)}" )
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

plt.scatter(y_test, y_test_pred_original, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect line
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Predicted vs Actual Yield")
plt.show()

# COMMAND ----------

plt.figure(figsize=(8, 4))
plt.scatter(
    y_test,
    y_test_pred_original,
    color='maroon',
 #   alpha=0.6,
    label='Predictions'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='orange',
    linestyle='--',
    linewidth=2,
    label='Perfect Prediction'
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Predicted vs Actual Yield")
plt.legend()
plt.grid(True)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Test loss is slightly lower than training loss. This indicates no overfitting, good generalisation and model learned meaningful structure. 
# MAGIC The trained network generalises well to unseen data, achieving a test MSE comparable to the training loss. This indicates that the model captures meaningful nonlinear relationships rather than memorising the training set.
# MAGIC
# MAGIC Training was performed using batch gradient descent, where gradients are computed over the entire training dataset and weights are updated once per epoch. The test set evaluation uses the same trained network architecture and weights, ensuring a fair assessment of generalisation
# MAGIC
# MAGIC ## Experiment 11: 
# MAGIC Network Architecture:  
# MAGIC
# MAGIC Input layer: 43 features; 
# MAGIC
# MAGIC 1 Hidden layer: [10, 20,30] neurons with sigmoid activation
# MAGIC
# MAGIC Output layer: 1 neuron with linear activation
# MAGIC
# MAGIC Loss: Mean Squared Error (MSE)
# MAGIC
# MAGIC Optimizer: Gradient Descent
# MAGIC
# MAGIC Learning rate: 0.01
# MAGIC
# MAGIC Epochs: 1000

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# -----------------------------
# 1. Network configuration
# -----------------------------
np.random.seed(42)

n_inputs = X_train.shape[1]
n_hidden_list = [10, 20, 30]  # first 10 neurons, then 20 neurons
n_output = 1
lr = 0.01
epochs = 1000

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Sigmoid activation function

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))  # Derivative of sigmoid activation function

# -----------------------------
# 2. Training loop for different hidden neurons
# -----------------------------
for n_hidden in n_hidden_list:
    # Initialize weights
    W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
    W2 = np.random.randn(n_output, n_hidden) *  np.sqrt(1. / n_hidden)

    loss_history = []

    for epoch in range(epochs):
        # --- Forward pass ---
        Z1 = np.dot(X_train, W1.T)
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2.T)
        y_pred = Z2  # linear output

        # --- Loss ---
        loss = np.mean((y_train_scaled - y_pred)**2)
        loss_history.append(loss)

        # --- Backpropagation ---
        dL_dy = 2 * (y_pred - y_train_scaled) / y_train_scaled.shape[0]
        dL_dW2 = np.dot(dL_dy.T, A1)
        dL_dA1 = np.dot(dL_dy, W2)
        dL_dZ1 = dL_dA1 * sigmoid_derivative(Z1)
        dL_dW1 = np.dot(dL_dZ1.T, X_train)

        # --- Update weights ---
        W1 -= lr * dL_dW1
        W2 -= lr * dL_dW2

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Hidden neurons: {n_hidden}, Epoch {epoch}, Loss: {loss:.4f}")

    # --- Plot training loss ---
    plt.plot(loss_history, label=f'{n_hidden} neurons')

plt.title("Training Loss for 1 Hidden Layer")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# COMMAND ----------

# -----------------------------
# 3. Test set evaluation
# -----------------------------

A1_test = sigmoid(np.dot(X_test, W1.T))
y_test_pred = np.dot(A1_test, W2.T)

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean

test_loss_original = np.mean((y_test - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

mean_yield = np.mean(y_test)
print(f"Mean y_test mean yield is {mean_yield} and std is {np.std(y_test)}" )
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC Training Behavior: The model showed steady but relatively slow convergence. For 10 neurons, loss decreased from 1.1324 at epoch 0 to 0.0135 at epoch 900, indicating successful learning but gradual improvement over epochs. Test Performance showed test RMSE (scaled): 0.118 with test RMSE (original units): 1091.42 kg and relative RMSE: 4.81%.  The model generalizes reasonably well but performs slightly worse compared to the tanh-based architecture tested earlier.
# MAGIC
# MAGIC The sigmoid activation function is capable of learning the regression mapping; however, due to weaker gradient flow and non-zero-centered outputs, it results in slower convergence and slightly higher prediction error compared to tanh. Thus, while sigmoid works, it is not the most efficient activation choice for this regression task.

# COMMAND ----------

import matplotlib.pyplot as plt

plt.scatter(y_test, y_test_pred_original, alpha=0.5)
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)], 'r--')  # perfect prediction line
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Predicted vs Actual Yield")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 12: 2 hidden layers with 20 and 10 neurons

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# -----------------------------
# 1. Network configuration
# -----------------------------
np.random.seed(42)

n_inputs = X_train.shape[1]  # 43 features
n_hidden1 = 20               # first hidden layer
n_hidden2 = 10               # second hidden layer
n_output = 1                 # yield prediction

# Randomly initialize weights
#W1 = np.random.randn(n_hidden1, n_inputs)    # input -> hidden1
#W2 = np.random.randn(n_hidden2, n_hidden1)  # hidden1 -> hidden2
#W3 = np.random.randn(n_output, n_hidden2)   # hidden2 -> output

W1 = np.random.randn(n_hidden1, n_inputs) * np.sqrt(1. / n_inputs)
W2 = np.random.randn(n_hidden2, n_hidden1) * np.sqrt(1. / n_hidden1)
W3 = np.random.randn(n_output, n_hidden2) * np.sqrt(1. / n_hidden2)


# Learning rate
lr = 0.01

# -----------------------------
# 2. Activation function
# -----------------------------
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# -----------------------------
# 3. Training loop
# -----------------------------
epochs = 1000
train_losses = []

for epoch in range(epochs):
    # --- Forward pass ---
    Z1 = np.dot(X_train, W1.T)      # (n_samples, n_hidden1)
    A1 = tanh(Z1)                    # hidden layer 1 activation

    Z2 = np.dot(A1, W2.T)            # (n_samples, n_hidden2)
    A2 = tanh(Z2)                    # hidden layer 2 activation

    Z3 = np.dot(A2, W3.T)            # (n_samples, 1)
    y_pred = Z3                       # linear output

    # --- Loss ---
    loss = np.mean((y_train_scaled - y_pred)**2)
    train_losses.append(loss)

    # --- Backpropagation ---
    # Step 1: dL/dy_pred
    dL_dy = 2 * (y_pred - y_train_scaled) / y_train_scaled.shape[0]   # (n_samples, 1)

    # Step 2: Output layer gradients
    dL_dW3 = np.dot(dL_dy.T, A2)                        # (1, n_hidden2)
    dL_dA2 = np.dot(dL_dy, W3)                          # (n_samples, n_hidden2)

    # Step 3: Hidden layer 2 gradients
    dL_dZ2 = dL_dA2 * tanh_derivative(Z2)              # (n_samples, n_hidden2)
    dL_dW2 = np.dot(dL_dZ2.T, A1)                      # (n_hidden2, n_hidden1)
    dL_dA1 = np.dot(dL_dZ2, W2)                        # (n_samples, n_hidden1)

    # Step 4: Hidden layer 1 gradients
    dL_dZ1 = dL_dA1 * tanh_derivative(Z1)             # (n_samples, n_hidden1)
    dL_dW1 = np.dot(dL_dZ1.T, X_train)                # (n_hidden1, n_inputs)

    # --- Update weights ---
    W1 -= lr * dL_dW1
    W2 -= lr * dL_dW2
    W3 -= lr * dL_dW3

    # --- Print loss every 100 epochs ---
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -----------------------------
# 5. Optional: Plot training loss
# -----------------------------
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss for 2 Hidden Layers")
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Layer	Neurons	Activation
# MAGIC Hidden 1,	20	neurons, tanh
# MAGIC
# MAGIC Hidden 2,	10 neurons,	tanh
# MAGIC
# MAGIC Output	1,	linear
# MAGIC
# MAGIC Objective: To see if adding depth improves the network’s ability to model nonlinear relationships between climate features and paddy yield. Rapid early decrease is seen. The network quickly learns major patterns in the first 100–300 epochs.
# MAGIC
# MAGIC Smooth convergence: Loss decreases steadily over 1000 epochs.
# MAGIC
# MAGIC Final loss (~0.86): Slightly lower than 1- hidden-layer networks with comparable total neurons (e.g., 30 neurons single hidden layer gave ~0.70 - 0.80 depending on activation), showing incremental improvement from depth. Depth helps modelling nonlinear interactions, but a wider single layer may already capture enough complexity.

# COMMAND ----------

# -----------------------------
# 4. Test loss
# -----------------------------
# Forward pass on X_test
Z1_test = np.dot(X_test, W1.T)
A1_test = tanh(Z1_test)
Z2_test = np.dot(A1_test, W2.T)
A2_test = tanh(Z2_test)
y_test_pred = np.dot(A2_test, W3.T)

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
rmse = np.sqrt(test_loss)  
print(f"Test MSE Loss: {test_loss} and Test RMSE Loss: {rmse}")

# COMMAND ----------

# MAGIC %md
# MAGIC Test loss is slightly higher than the best 1-layer sigmoid network. This is expected as, although depth adds capacity,  small datasets may not benefit much from extra layers unless regularisation or more data is used.

# COMMAND ----------


y_test_pred_original = y_test_pred * y_std + y_mean
y_test_original = y_test 

test_loss_original = np.mean((y_test_original - y_test_pred_original)**2)

rmse_original = np.sqrt(test_loss_original)  

# Compute relative RMSE
mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE Loss (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE Loss (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 13: Linear (hidden + Output)

# COMMAND ----------


# Select continuous columns to standardize (0–25)
cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

# =========================
# EXPERIMENT 9
# lINEAR hidden neuron
# Regression output (paddy yield)
# =========================
# -----------------------------
# 1. Network configuration
# -----------------------------
np.random.seed(42)

n_inputs = X_train.shape[1]
n_hidden_list = [10, 20, 30]  # first 10 neurons, then 20 neurons
n_output = 1
lr = 0.01
epochs = 1000

# Activation function
def linear(x):
    return x
def linear_derivative(x):
    return np.ones_like(x)

# -----------------------------
# 2. Training loop for different hidden neurons
# -----------------------------
for n_hidden in n_hidden_list:
    # Initialize weights
    W1 = np.random.randn(n_hidden, n_inputs) * np.sqrt(1. / n_inputs)
    W2 = np.random.randn(n_output, n_hidden) * np.sqrt(1. / n_hidden)

    loss_history = []

    for epoch in range(epochs):
        # --- Forward pass ---
        Z1 = np.dot(X_train, W1.T)
        A1 = (Z1)
        Z2 = np.dot(A1, W2.T)
        y_pred = Z2  # linear output

        # --- Loss ---
        loss = np.mean((y_train_scaled - y_pred)**2)
        loss_history.append(loss)

        # --- Backpropagation ---
        dL_dy = 2 * (y_pred - y_train_scaled) / y_train_scaled.shape[0]
        dL_dW2 = np.dot(dL_dy.T, A1)
        dL_dA1 = np.dot(dL_dy, W2)
        dL_dZ1 = dL_dA1 * linear_derivative(Z1)
        dL_dW1 = np.dot(dL_dZ1.T, X_train)

        # --- Update weights ---
        W1 -= lr * dL_dW1
        W2 -= lr * dL_dW2

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Hidden neurons: {n_hidden}, Epoch {epoch}, Loss: {loss:.4f}")

    # --- Plot training loss ---
    plt.plot(loss_history, label=f'{n_hidden} neurons')

plt.title("Training Loss for 1 Hidden Layer")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# COMMAND ----------

# TESTING ON THE X_TEST
Z1_test = np.dot(X_test, W1.T)
A1_test = (Z1_test)
Z2_test = np.dot(A1_test, W2.T)
y_test_pred = Z2_test

test_loss = np.mean((y_test_scaled - y_test_pred)**2)
print(f"Test Loss: {test_loss:.4f}")
RMSE = np.sqrt(test_loss)
print(f"Test RMSE Loss: {RMSE}")

# testing on the original data:
y_test_pred_original = y_test_pred * y_std + y_mean
y_test_original = y_test 
test_loss_original = np.mean((y_test_original - y_test_pred_original)**2)
print("Test MSE Loss (original units):", test_loss_original)

rmse_original = np.sqrt(test_loss_original)  
print(f"Test RMSE Loss (original units: {rmse_original}")

# Compute relative RMSE
mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE Loss (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE Loss (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

print("y_test_pred_original min, max, mean:", y_test_pred_original.min(), y_test_pred.max(), y_test_pred.mean())

# COMMAND ----------


y_test_pred

# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 13 B: 
# MAGIC Reducing the Learning rate to 0.001
# MAGIC The results show increase of the training MSE, test MSE and RMSE relative performance of 5.92% in comparision to the previous experiments

# COMMAND ----------

pip install tensorflow


# COMMAND ----------

# MAGIC %md
# MAGIC # Experiment 14: TensorFlow/Keras:  Tanh with 30 Neurons in one layer and output activation: linear
# MAGIC
# MAGIC #### Test MSE Loss: 0.0099
# MAGIC #### Test RMSE: 0.0996
# MAGIC
# MAGIC ####  1/18 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
# MAGIC #### 18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 
# MAGIC ####  Test MSE Loss (original units): 848112.28 KG
# MAGIC ####  Test RMSE Loss (original units): 920.93 KG
# MAGIC ####  Relative RMSE (% of mean yield): 4.05%

# COMMAND ----------

# t continuous columns to standardize (0–25)
cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# -----------------------------
# 2. Build Model
# -----------------------------
model = Sequential([
    Dense(30, input_dim=X_train.shape[1], activation='tanh'),  # Hidden layer with 30 neurons
    Dense(1, activation='linear')                               # Output layer
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse')

# -----------------------------
# 3. Train Model
# -----------------------------
history = model.fit(X_train, y_train_scaled, epochs=1000, batch_size=32,
                    validation_data=(X_test, y_test_scaled), verbose=0)

# -----------------------------
# 4. Plot Training & Validation Loss
# -----------------------------
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Test Loss')
plt.legend()
plt.show()

# -----------------------------
# 5. Evaluate on Test Data
# -----------------------------
test_loss = model.evaluate(X_test, y_test_scaled, verbose=0)
print(f"Test MSE Loss: {test_loss:.4f}")
rmse = np.sqrt(test_loss)
print(f"Test RMSE: {rmse:.4f}")

# -----------------------------
# 5. Predict on Test Data and Convert to Original Units
# -----------------------------
y_test_pred_scaled = model.predict(X_test)
y_test_pred_original = y_test_pred_scaled * y_std + y_mean   # Rescale back to original units

# Compute loss in original units
test_loss_original = np.mean((y_test - y_test_pred_original)**2)
rmse_original = np.sqrt(test_loss_original)

# Compute relative RMSE
mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE Loss (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE Loss (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Differences between the from-scratch model and Keras model
# MAGIC
# MAGIC Architecture : Same architecture for from scratch and keras model. So the mathematical model is the same.
# MAGIC The main differences are not in architecture ((4.09% vs 4.32%)) they are in optimization and training mechanics.
# MAGIC
# MAGIC 1. Optimizer: In manual implementation, plain Gradient Descent with fixed learning rate = 0.01 is being used. It also uses full dataset each step. But in Keras implementation, Adam optimizer is used. This uses adaptive learning rates, uses momentum and adjusts updates per parameter, which converges faster. Escapes flat regions better. Adam is simply more advanced than vanilla gradient descent.
# MAGIC
# MAGIC 2. Mini-batch Training: In manual version, full-batch gradient descent is used. But in Keras, batch_size=32 is used (mini-batch gradient descent). This mini batch training adds controlled noise to gradients often improves generalization and reduce overfitting. Helps escape poor local minima, which can improve RMSE slightly.
# MAGIC
# MAGIC 3. Numerical Stability & Implementation Efficiency: In  TensorFlow, highly optimized, uses stable matrix operations and better floating-point handling and efficient memory management. In NumPy version, correct mathematically, not optimized at deep framework level. This can cause small differences in convergence quality.
# MAGIC
# MAGIC 4. Weight Initialization Subtleties: Even though both use Xavier-style scaling, TensorFlow's Dense layer may apply slightly different initialization logic. Adam interacts differently with initial weights. Small initialization differences can cause different local minimum.
# MAGIC
# MAGIC 5. Results: Manual (1 hidden tanh)	4.32%;  Manual (2 hidden tanh)	4.51%, keras (tanh + Adam)	4.09%. The difference is small. This shows that the manual math implementation is correct, gradients are correct and understanding is solid. The improvement comes from better optimization, not a different model.
# MAGIC
# MAGIC # Experiment 15: TensorFlow/Keras (Sigmoid in the hidden layer and linear in the output layer)

# COMMAND ----------

cont_cols = X_train.columns[0:26]   # numeric continuous features
other_cols = X_train.columns[26:]   # sin/cos + one-hot categorical

# Compute mean and std only for continuous columns
X_mean = X_train[cont_cols].mean()
X_std = X_train[cont_cols].std()

# Standardize continuous columns
X_train_scaled_cont = (X_train[cont_cols] - X_mean) / X_std
X_test_scaled_cont  = (X_test[cont_cols]  - X_mean) / X_std

# Keep other columns unchanged
X_train_other = X_train[other_cols]
X_test_other  = X_test[other_cols]

# Combine standardized continuous + unchanged columns
X_train= pd.concat([X_train_scaled_cont, X_train_other], axis=1)
X_test  = pd.concat([X_test_scaled_cont, X_test_other], axis=1)

# computing the mean and std for y_train 
y_mean = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

# standardising the y_train and y_test
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# y_train may be pandas Series
y_train_scaled = np.asarray(y_train_scaled).reshape(-1, 1)

# y_test_scaled is already numpy
y_test_scaled = np.asarray(y_test_scaled).reshape(-1, 1)

if isinstance(y_test, pd.Series):
    y_test = y_test.to_numpy().reshape(-1,1)
else:
    y_test = y_test.reshape(-1,1)

# COMMAND ----------

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 2. Define the model
# -----------------------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # 43 features
    Dense(20, activation='sigmoid'),          # Hidden layer, 20 neurons
    Dense(1, activation='linear')             # Output layer, linear
])

# -----------------------------
# 3. Compile the model
# -----------------------------
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='mse',
              metrics=['mse'])

# -----------------------------
# 4. Train the model
# -----------------------------
history = model.fit(X_train, y_train_scaled,
                    epochs=500,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=2)

# -----------------------------
# 5. Evaluate on test set
# -----------------------------
test_loss, test_mse = model.evaluate(X_test, y_test_scaled, verbose=0)
print("\nTest MSE Loss (scaled y):", test_mse)

rmse = np.sqrt(test_mse)
print(f"Test RMSE Loss (scaled y): {rmse:.4f}")

# Convert predictions back to original scale
y_test_pred = model.predict(X_test)

# COMMAND ----------

y_test_pred_original = y_test_pred * y_std + y_mean
y_test_original = y_test

test_loss_original = np.mean((y_test_original - y_test_pred_original)**2)
print("Test MSE Loss (original units):", test_loss_original)

rmse_original = np.sqrt(test_loss_original)

# Compute relative RMSE
mean_yield = np.mean(y_test)
relative_rmse_percent = (rmse_original / mean_yield) * 100

# Print results
print(f"Test MSE Loss (original units): {test_loss_original:.2f} KG")
print(f"Test RMSE Loss (original units): {rmse_original:.2f} KG")
print(f"Relative RMSE (% of mean yield): {relative_rmse_percent:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC The starting loss is already very small at Epoch 1 (0.2822 quickly drops to ~0.013). Because OF Adam optimizer, Mini-batch training (batch_size default or specified), better weight initialization.  In Keras, one epoch = many mini-batch updates. If there are 63 steps per epoch, it means 63 gradient updates happen in 1 epoch. So by the time Epoch 1 finishes, the model has already updated weights 63 times. In manual NumPy version, 1 epoch = one update (full batch). Much slower learning per epoch. So the comparison is not 1-to-1. Adam and this mini-batch leads to very aggressive early optimization. Adam uses momentum, uses adaptive learning rates. So loss drops dramatically in first few epochs.

# COMMAND ----------

# MAGIC %md
# MAGIC # References:
# MAGIC # GitHub account:
# MAGIC https://github.com/research-selvi-datascience/Mathematics_CA2_ANN_Implementation/blob/main/CA2_Mathematics_ANN.ipynb 
# MAGIC
# MAGIC 1. GeeksforGeeks (2024). Building Artificial Neural Networks (ANN) from Scratch. [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/deep-learning/building-artificial-neural-networks-ann-from-scratch/.
# MAGIC 2. https://www.facebook.com/jason.brownlee.39 (2016). How to Code a Neural Network with Backpropagation In Python. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/.
# MAGIC 3.  Uci.edu. (2025). UCI Machine Learning Repository. [online] Available at: https://archive.ics.uci.edu/dataset/1186/paddy+dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC