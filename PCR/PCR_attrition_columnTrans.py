# Data cleaning and preparation
# ==============================================================================
import multiprocessing

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

# Graphs
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

# Accuracy tests
# ==============================================================================
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# Preprocessing and modeling
# ==============================================================================
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Warnings configuration
# ==============================================================================
style.use('ggplot') or plt.style.use('ggplot')

### EXTRACTING DATA INPUT ###
# ==============================================================================
data = None
try:
    data = pd.read_csv(
        'file:/Users/lorena/Public/knime-workspace/Datos%20para%20pra%CC%81cticas/Estadistica%20Bloque%20I/WA_Fn-UseC_-HR-Employee-Attrition.csv')
except IOError:
    print("Cannot open the csv file")

### TRANSFORMING CATEGORICAL DATA TO NUMERIC ###
# ==============================================================================

# 1. Filtering columns to encode categorical data
# ==============================================================================

df = data.drop(columns=['Over18', 'EmployeeNumber', 'EmployeeCount', 'StandardHours'])

X = df.drop(columns='Attrition')


def parse_string_column_to_numeric(dataframe, column_name: str):
    df_dummy = pd.get_dummies(dataframe['{}'.format(column_name)])

    first_column_name = df_dummy.columns[0]
    df_dummy.rename(columns={'{}'.format(first_column_name): '{}'.format(column_name)}, inplace=True)

    second_column_name = df_dummy.columns[1]
    df_dummy.drop(columns={'{}'.format(second_column_name)}, inplace=True)
    return df_dummy.values.ravel()


y = parse_string_column_to_numeric(df, 'Attrition')

# 2. Select categorical columns (strings) and numerical columns(int, floats)
# ==============================================================================

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(X)
categorical_columns = categorical_columns_selector(X)

# 3. Create our ColumnTransfomer by specifying three values: the preprocessor name, the transformer, and the columns.
# First, let’s create the preprocessors for the numerical and categorical parts.
# ==============================================================================

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

# 4. Create the transformer and associate each of these preprocessors with their respective columns
# ==============================================================================

# NOTE: ColumnsTransformer will internally call fit_transform or transform!
preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

df_prepro = preprocessor.fit_transform(X)

names_columns_preprocessor = preprocessor.get_feature_names_out()  # Column's names already changed in preprocessor step(they aren't X columns anymore)

df_scaled_named = pd.DataFrame(data=df_prepro,
                               columns=names_columns_preprocessor)  # Df with scaled and transformed values + new transformed columns names

### ADECUACCY TEST FOR PCA ###

# 1. Kaiser-Meyer-Olkin (KMO) Test
# ==============================================================================

kmo_all, kmo_model = calculate_kmo(df_scaled_named)
print(f"Kaiser-Meyer-Olkin (KMO) Test is: {kmo_model}")
print(
    kmo_model)  # This value indicates that you shouldn't proceed with your planned factor analysis because it's miserable.

# 2. Bartlett’s Test
# ==============================================================================

chi_square_value, p_value = calculate_bartlett_sphericity(df_scaled_named)
print(
    f"chi_square_value, p_value is:{chi_square_value, p_value}")


# P-value is less than 0.05, we reject HO and conclude that correlation is present among the variables which is a green signal to apply factor analysis.


### CORRELATION MATRIX###
# ==============================================================================

def tidy_corr_matrix(corr_mat):
    """
    Función para convertir una matriz de correlación de pandas en formato tidy
    """
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1', 'variable_2', 'r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)

    return corr_mat


corr_matrix = df_scaled_named.select_dtypes(include=['float64', 'int']).corr(method='pearson')
print('corr matrix')
print(tidy_corr_matrix(corr_matrix).head(15))

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap='BrBG', fmt='.2f')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
plt.show()

### MODEL ###
# ==============================================================================

# 1. Splitting our data into train and test sets
# ==============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=1234,
    shuffle=True)

# 2. Creating and training the model
# ==============================================================================
pipe_model = make_pipeline(preprocessor, PCA(), LogisticRegression())
trained = pipe_model.fit(X=X_train, y=y_train)
print('X_test.head()')
print(X_test.head())

# 3. Predictions
# ==============================================================================
predictions = pipe_model.predict(X=X_test)
predictions = predictions.flatten()

# 4. Accuracy score
# ==============================================================================
print(pipe_model.score(X_test, y_test))

# 5. RMSE
# ==============================================================================
rmse_ols = mean_squared_error(
    y_true=y_test,
    y_pred=predictions,
    squared=False)
print("")
print(f"El error (rmse) de test es: {rmse_ols}")

### CROSS VALIDATION-PCA ###
# ==============================================================================

# 1. Grid Search Cross-Validation for Hyperparameter Tuning
# ==============================================================================

param_grid = {'pca__n_components': [1, 2, 4, 6, 8, 10, 15]}

grid = GridSearchCV(
    estimator=pipe_model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    n_jobs=multiprocessing.cpu_count() - 1,
    cv=KFold(n_splits=5, random_state=123, shuffle=True),
    refit=True,
    verbose=0,
    return_train_score=True)

grid.fit(X_train, y_train)

# 2. Results
# ==============================================================================
results = pd.DataFrame(grid.cv_results_)
results.filter(regex='(param.*|mean_t|std_t)') \
    .drop(columns='params') \
    .sort_values('mean_test_score', ascending=False) \
    .head(3)

print(results)

## 3. Plotting cross validation per parameter
# ==============================================================================

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.84), sharey=True)

results.plot('param_pca__n_components', 'mean_train_score', ax=ax)
results.plot('param_pca__n_components', 'mean_test_score', ax=ax)
ax.fill_between(results.param_pca__n_components.astype(float),
                results['mean_train_score'] + results['std_train_score'],
                results['mean_train_score'] - results['std_train_score'],
                alpha=0.2)
ax.fill_between(results.param_pca__n_components.astype(float),
                results['mean_test_score'] + results['std_test_score'],
                results['mean_test_score'] - results['std_test_score'],
                alpha=0.2)
ax.legend()
ax.set_title('Evolución del error CV')
ax.set_ylabel('neg_root_mean_squared_error')
plt.show()

# 4. Best hyperparamets
# ==============================================================================

print("----------------------------------------")
print("Best hyperparameters founded (cv)")
print("----------------------------------------")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

# cross validation results show the best model has the first 30 components, however from 15 components we dont obtain better results


# 5. Cumulative explained variance ratio
# ==============================================================================

# 6. Extract the trained model from pipeline
modelo_pca = pipe_model.named_steps['pca']  # extracting pca step

prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()

print('------------------------------------------')
print('Porcentaje de varianza explicada acumulada')
print('------------------------------------------')
print(prop_varianza_acum)

# 7. Plotting cumulative explained variance ratio
explained = modelo_pca.explained_variance_ratio_
explained = explained[0:30]
plt.plot(np.cumsum(explained))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

### MODEL: TRAINING LOGISTIC REGRESSION MODEL BASED ON 24 COMPONENTS ###
# ==============================================================================


pipe_model2 = make_pipeline(preprocessor, PCA(n_components=24), LogisticRegression())
pipe_model2.fit(X=X_train, y=y_train)

# 1.Predictions
# ==============================================================================

predictions_24 = pipe_model2.predict(X=X_test)
predictions_24 = predictions_24.flatten()

# 2. RMSE
# ==============================================================================

rmse_pcr = mean_squared_error(y_true=y_test, y_pred=predictions_24, squared=False)
print(f'RMSE for 24 components is:{rmse_pcr}')

### INTERPRETACION ###
# ==============================================================================

# 1. Transforming array to DF adding names to axis
# ==============================================================================

modelo_pca2 = pipe_model2.named_steps['pca']  # extracting PCA trained model step

df_components = pd.DataFrame(data=modelo_pca2.components_, columns=df_scaled_named.columns,
                             index=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11',
                                    'PC12', 'PC13',
                                    'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23',
                                    'PC24'])

# 2. Visualizing the influence of variables on each component
# ==============================================================================

sns.heatmap(df_components)
plt.figure(figsize=(100, 600))
heatmap = sns.heatmap(df_components, vmin=-1, vmax=1, annot=True, cmap='BrBG', fmt='.1f')
heatmap.set_title('Heatmap', fontdict={'fontsize': 18}, pad=12)
plt.show()

# 3. Final interpretation
# ==============================================================================

## The analysis indicates our variables are not highly correlated from the beginninh, it means our PCR analysis don't show good results(i.e: RMSE increase after PCR) Even KMO test showed we shouldn't proceed with  PCA analysis because it's bad)
## Describing some components from de PCA:
# PC1 combinations :JobLevel, Yearsatcompnay, Totalworkingyears
# PC2: Age and Number of companies worked
# PC3: Salary hike and Performance rating
# PC4: Stock options level
# ...
