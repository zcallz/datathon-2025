import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
import optuna

# -------------------------
# 1️⃣ Veri Yükleme
# -------------------------
train_path = r"C:\Users\ozlemcal\Downloads\train.csv"
test_path = r"C:\Users\ozlemcal\Downloads\test.csv"
print("[INFO] Veriler yükleniyor...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# -------------------------
# 2️⃣ Gelişmiş Özellik Mühendisliği
# -------------------------
def create_session_features(df):
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df = df.sort_values(by=['user_session', 'event_time'])

    session_feat = df.groupby('user_session').agg(
        total_events=('product_id', 'count'),
        unique_products=('product_id', 'nunique'),
        unique_categories=('category_id', 'nunique'),
        min_time=('event_time', 'min'),
        max_time=('event_time', 'max')
    )
    
    event_counts = pd.crosstab(df['user_session'], df['event_type'])
    session_feat = session_feat.merge(event_counts, left_index=True, right_index=True, how='left')
    
    session_feat['session_duration_sec'] = (session_feat['max_time'] - session_feat['min_time']).dt.total_seconds()
    session_feat['events_per_sec'] = session_feat['total_events'] / session_feat['session_duration_sec'].replace(0, 1)

    session_feat['view_to_cart_rate'] = session_feat.get('add_to_cart', 0) / (session_feat.get('view', 0) + 1)
    session_feat['cart_to_buy_rate'] = session_feat.get('buy', 0) / (session_feat.get('add_to_cart', 0) + 1)
    
    session_feat['product_diversity'] = session_feat['unique_products'] / session_feat['total_events']
    session_feat['category_diversity'] = session_feat['unique_categories'] / session_feat['total_events']
    
    session_feat['start_hour'] = session_feat['min_time'].dt.hour
    session_feat['start_day_of_week'] = session_feat['min_time'].dt.dayofweek
    session_feat['start_month'] = session_feat['min_time'].dt.month
    
    if 'session_value' in df.columns:
        session_value = df.groupby('user_session')['session_value'].mean()
        session_feat['session_value'] = session_value

    session_feat = session_feat.reset_index()
    
    user_agg_more = df.groupby('user_id').agg(
        user_total_sessions=('user_session', 'nunique'),
        user_avg_events=('product_id', 'count')
    )
    user_agg_more['user_avg_events'] = user_agg_more['user_avg_events'] / user_agg_more['user_total_sessions']
    user_agg_more.drop(columns=['user_total_sessions'], inplace=True)
    
    user_counts = df.groupby('user_id')['event_type'].value_counts().unstack(fill_value=0).reset_index()
    user_counts.columns = [f'user_{col}_count' if col != 'user_id' else col for col in user_counts.columns]
    
    session_feat = session_feat.merge(df[['user_session', 'user_id']].drop_duplicates(), on='user_session', how='left')
    session_feat = session_feat.merge(user_agg_more, on='user_id', how='left')
    session_feat = session_feat.merge(user_counts, on='user_id', how='left')

    # Yeni: Lag (geçmiş) tabanlı özellikler
    user_session_list = session_feat.groupby('user_id')['user_session'].apply(list)
    
    prev_session_value = {}
    for user, sessions in user_session_list.items():
        for i, session_id in enumerate(sessions):
            if i > 0 and 'session_value' in session_feat.columns:
                prev_session_value[session_id] = session_feat.loc[session_feat['user_session'] == sessions[i-1], 'session_value'].values[0]
            else:
                prev_session_value[session_id] = np.nan
    
    prev_session_df = pd.DataFrame(prev_session_value.items(), columns=['user_session', 'prev_session_value'])
    session_feat = session_feat.merge(prev_session_df, on='user_session', how='left')

    session_feat.drop(columns=['min_time', 'max_time', 'user_id'], inplace=True)
    return session_feat

train_feats = create_session_features(train)
test_feats = create_session_features(test)

for col in ['session_duration_sec', 'view_to_cart_rate', 'cart_to_buy_rate', 'events_per_sec', 'product_diversity', 'category_diversity', 'user_avg_events', 'prev_session_value']:
    if col in train_feats.columns:
        median_val = train_feats[col].median()
        train_feats[col] = train_feats[col].fillna(median_val)
        test_feats[col] = test_feats[col].fillna(median_val)

train_feats['session_duration_sec'] = np.log1p(train_feats['session_duration_sec'].apply(lambda x: max(0, x)))
test_feats['session_duration_sec'] = np.log1p(test_feats['session_duration_sec'].apply(lambda x: max(0, x)))

print("[INFO] Özellik mühendisliği tamamlandı.")

# -------------------------
# 3️⃣ Model Optimizasyonu & Stacking
# -------------------------
target = 'session_value'
features = [c for c in train_feats.columns if c not in ['user_session', target]]

X_train = train_feats[features]
y_train = train_feats[target]
X_test = test_feats[features]

# LGBM için Optuna ile parametre optimizasyonu (100 deneme)
def objective_lgb(trial):
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 30, 80),
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'random_state': 42
    }
    model = lgb.LGBMRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return -score

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=100, show_progress_bar=True)
best_lgb_params = study_lgb.best_params
print(f"LGBM En iyi parametreler: {best_lgb_params}")

# XGBoost için Optuna ile parametre optimizasyonu (100 deneme)
def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return -score

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=100, show_progress_bar=True)
best_xgb_params = study_xgb.best_params
print(f"XGBoost En iyi parametreler: {best_xgb_params}")

# CatBoost için Optuna ile parametre optimizasyonu (75 deneme)
def objective_cat(trial):
    params = {
        'objective': 'RMSE',
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 1),
        'random_seed': 42
    }
    model = cb.CatBoostRegressor(**params, silent=True)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return -score

study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=75, show_progress_bar=True)
best_cat_params = study_cat.best_params
print(f"CatBoost En iyi parametreler: {best_cat_params}")

# Nihai Modelleri Eğitme
lgb_model = lgb.LGBMRegressor(**best_lgb_params)
xgb_model = xgb.XGBRegressor(**best_xgb_params)
cat_model = cb.CatBoostRegressor(**best_cat_params, silent=True)

print("[INFO] Nihai modeller eğitiliyor...")
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)

# Stacking için OOF (Out-of-Fold) Tahminleri
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("[INFO] OOF tahminleri oluşturuluyor...")
lgb_oof = cross_val_predict(lgb_model, X_train, y_train, cv=kf, n_jobs=-1)
xgb_oof = cross_val_predict(xgb_model, X_train, y_train, cv=kf, n_jobs=-1)
cat_oof = cross_val_predict(cat_model, X_train, y_train, cv=kf, n_jobs=-1)

# Meta-model eğitimi
meta_model = LinearRegression()
meta_model.fit(pd.DataFrame({'lgb_oof': lgb_oof, 'xgb_oof': xgb_oof, 'cat_oof': cat_oof}), y_train)

# Test verisi üzerinde nihai tahminler
lgb_pred = lgb_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
cat_pred = cat_model.predict(X_test)

final_pred = meta_model.predict(pd.DataFrame({'lgb_oof': lgb_pred, 'xgb_oof': xgb_pred, 'cat_oof': cat_pred}))

print("[INFO] Ensemble tahmini tamamlandı.")

# -------------------------
# 4️⃣ Gönderim Dosyasını Oluştur
# -------------------------
submission_df = pd.DataFrame({
    'user_session': test_feats['user_session'],
    'session_value': final_pred
})

submission_df = submission_df.groupby('user_session')['session_value'].mean().reset_index()
submission_df['session_value'] = submission_df['session_value'].apply(lambda x: max(0, x)).round(2)

submission_df.to_csv('submission_final_advanced.csv', index=False)
print("[INFO] Gönderim dosyası başarıyla oluşturuldu: submission_final_advanced.csv")