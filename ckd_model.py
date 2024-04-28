import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load Dataset
url = "Chronic_Kidney_Disease_DataSet_UCI/ckd.csv"
df = pd.read_csv(url, na_values='?')

# Check for missing values
print(df.isnull().sum())

print()

# Cenverting necessary coluns to nurmaric type
df['pcv'] = pd.to_numeric(df['pcv'], errors="coerce")
df['wbcc'] = pd.to_numeric(df['wbcc'], errors="coerce")
df['rbcc'] = pd.to_numeric(df['rbcc'], errors="coerce")

print(df.info())

# Preprocessing
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

print(numerical_cols)
print(categorical_cols)
print()


# Categorical data check unique values
def check_unique_categorical(categorical):
    print('Categorical data check unique values')
    for col in categorical:
        print(f'{col} has {df[col].unique()}')


check_unique_categorical(categorical_cols)

# fixed \t
df['dm'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
df['cad'].replace(to_replace='\tno', value='no', inplace=True)
df['class'].replace(to_replace="ckd\t", value="ckd", inplace=True)

print('\nAfter fixed')
check_unique_categorical(categorical_cols)


# Plotting
def display_plt():
    plt.figure(figsize=(20, 15))
    plt_number = 1
    for col in numerical_cols:
        if plt_number <= 14:
            ax = plt.subplot(3, 5, plt_number)
            sns.distplot(df[col])
            plt.xlabel(col)
        plt_number += 1
    plt.tight_layout()
    plt.show()


# Plotting distribution data
# display_plt()


# handel data skewness
def handel_skewness(col):
    df[col] = np.log1p(df[col])


handel_skewness('bu')
handel_skewness('sod')
handel_skewness('pot')
handel_skewness('sc')
handel_skewness('su')

# After data skewness
# display_plt()

print()
print(df.isna().sum())


# filling null values, we will use two methods, random sampling for higher null values and mean/mode sampling for lower null values
def random_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample


def mean_imputation(feature):
    df[feature] = df[feature].fillna(df[feature].mean())


def mode_imputation(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)


# Filling numerical columns null values using mean_imputation
for col in numerical_cols:
    mean_imputation(col)

print('\nAfter filling numerical columns null values')
print(df[numerical_cols].isnull().sum())

# Filling categorical columns null values using mode_imputation
for col in categorical_cols:
    mode_imputation(col)

print('\nAfter filling categorical columns null values')
print(df[categorical_cols].isnull().sum())

# Perform one-hot encoding for categorical columns
# df = pd.get_dummies(df, columns=categorical_cols)

# categorical data label encoding
encode = LabelEncoder()
for col in categorical_cols:
    df[col] = encode.fit_transform(df[col])

# Scale numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# data splitting
X = df.drop(columns='class', axis=1)
y = df['class']
# y = df['class'].map({'ckd': 1, 'notckd': 0})    # Assuming 'ckd' as positive class

print(df.head())

df.to_csv('updated.csv', index=False)

# Splitting dataset for training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=31,
    batch_size=32,
    validation_split=0.2
)


# Plot the Training Results
def plot_result(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]

    for idx, metrics in enumerate(metrics):
        ax.plot(metrics, color=color[idx])

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, 30])
    plt.ylim(ylim)

    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.close()


# Retrieve training result
train_loss = history.history["loss"]
train_acc = history.history["accuracy"]
valid_loss = history.history["val_loss"]
valid_acc = history.history["val_accuracy"]

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

plot_result(
    [train_loss, valid_loss],
    ylabel="Loss",
    ylim=[0.0, 5.0],
    metric_name=['Training Loss', 'Validation Loss'],
    color=['g', 'b']
)

plot_result(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.0, 2.0],
    metric_name=['Training Accuracy', 'Validation Accuracy'],
    color=['g', 'b']
)

print(model.summary())

# Predictions
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype('int32')

# Compute evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
accuracy_score = accuracy_score(y_test, y_pred_binary)

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Confusion Matrix:")
print(conf_matrix)
print('Precision: ', precision)
print('Recall:', recall)
print('F1-score:', f1)
print('accuracy:', accuracy_score)

print('Classification Report:')
print(classification_report(y_test, y_pred_binary))
