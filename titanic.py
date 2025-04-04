import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_prepare_data(train_path='train.csv', test_path='test.csv'):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    test_passenger_id = test_data['PassengerId']
    
    y = train_data['Survived']
    
    return train_data, test_data, y, test_passenger_id

def engineer_features(train_data, test_data):
    train_data['is_train'] = 1
    test_data['is_train'] = 0
    combined = pd.concat([train_data, test_data], axis=0)
    
    combined = combined.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
    
    combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    combined['Title'] = combined['Title'].map(lambda x: title_mapping.get(x, 'Rare'))
    
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    combined['FareBin'] = pd.qcut(combined['Fare'].fillna(combined['Fare'].median()), 4, labels=[0, 1, 2, 3])
    
    combined['AgeBin'] = pd.cut(combined['Age'].fillna(combined['Age'].median()), 5, labels=[0, 1, 2, 3, 4])
    
    combined = combined.drop(['Name', 'Age', 'Fare'], axis=1)
    
    train_processed = combined[combined['is_train'] == 1].drop('is_train', axis=1)
    test_processed = combined[combined['is_train'] == 0].drop(['is_train', 'Survived'], axis=1)
    
    return train_processed, test_processed

def create_preprocessor():
    numeric_features = ['SibSp', 'Parch', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'FareBin', 'AgeBin']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def build_ensemble_model():
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft'  # Use probabilities for voting
    )
    
    return ensemble

def train_and_tune_model(X, y, preprocessor):
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    rf_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    gb_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
    
    lr_params = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']
    }
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest...")
    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_val_score = accuracy_score(y_val, best_rf.predict(X_val))
    print(f"RF Validation Score: {rf_val_score:.4f}")
    
    print("Training Gradient Boosting...")
    gb_grid = GridSearchCV(gb_pipeline, gb_params, cv=5, scoring='accuracy')
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_
    gb_val_score = accuracy_score(y_val, best_gb.predict(X_val))
    print(f"GB Validation Score: {gb_val_score:.4f}")
    
    print("Training Logistic Regression...")
    lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=5, scoring='accuracy')
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    lr_val_score = accuracy_score(y_val, best_lr.predict(X_val))
    print(f"LR Validation Score: {lr_val_score:.4f}")
    
    best_rf_model = best_rf.named_steps['classifier']
    best_gb_model = best_gb.named_steps['classifier']
    best_lr_model = best_lr.named_steps['classifier']
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', best_rf_model),
            ('gb', best_gb_model),
            ('lr', best_lr_model)
        ],
        voting='soft'
    )
    
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])
    
    print("Training final ensemble...")
    final_pipeline.fit(X, y)
    
    return final_pipeline, best_rf, best_gb, best_lr

def titanic_prediction():
    print("Loading data...")
    train_data, test_data, y, test_passenger_id = load_and_prepare_data()
    
    print("Engineering features...")
    X_processed, test_processed = engineer_features(train_data.copy(), test_data.copy())
    
    print("Creating preprocessor...")
    preprocessor = create_preprocessor()
    
    print("Training models...")
    ensemble_model, rf_model, gb_model, lr_model = train_and_tune_model(X_processed, y, preprocessor)
    
    print("Making predictions...")
    predictions = ensemble_model.predict(test_processed)
    
    submission = pd.DataFrame({
        'PassengerId': test_passenger_id,
        'Survived': predictions
    })
    submission.to_csv('ensemble_submission.csv', index=False)
    print("Submission file created: ensemble_submission.csv")
    
    print("\nModel Performance Analysis:")
    for name, model in [("Random Forest", rf_model), ("Gradient Boosting", gb_model), ("Logistic Regression", lr_model)]:
        y_pred = model.predict(X_processed)
        accuracy = accuracy_score(y, y_pred)
        print(f"{name} Accuracy on training data: {accuracy:.4f}")
    
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(ensemble_model, X_processed, y, cv=5, scoring='accuracy')
    print(f"\nEnsemble 5-fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return ensemble_model, rf_model, gb_model, lr_model

# Run the prediction
if __name__ == "__main__":
    ensemble_model, rf_model, gb_model, lr_model = titanic_prediction()