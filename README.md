# datathon-2025

Bu metin, projenin amacını, kullanılan yöntemleri ve her bir tekniğin neden bu kadar etkili olduğunu potansiyel işverenlere ve diğer veri bilimcilere açıkça gösterecektir.
---
# TR
# Veri Bilimi Yarışması Projesi: Kullanıcı Seansı Değeri Tahmini

Bu proje, bir e-ticaret platformunda kullanıcı seanslarının değerini tahmin etmeyi amaçlayan bir veri bilimi yarışması için geliştirilmiştir. Elde edilen 540'lık MSE (Ortalama Kare Hata) değeri, modelin yüksek doğrulukta tahminler yaptığını göstermektedir. guncelsonkod.py için hazırlanmıştır.

---

# Kullanılan Yaklaşım ve Teknikler

Yüksek bir skor elde etmek için basit bir model yerine, profesyonel seviyede kullanılan aşağıdaki ileri veri bilimi teknikleri birleştirilmiştir:

* **Gelişmiş Özellik Mühendisliği:** Veriye dayalı güçlü sinyaller oluşturarak modelin daha iyi öğrenmesini sağlama.
* **Üçlü Ensemble ve Stacking:** Birden fazla güçlü modelin tahminlerini birleştirerek daha kararlı ve doğru bir nihai tahmin elde etme.
* **Otomatik Hiperparametre Optimizasyonu:** Modellerin en iyi performansını yakalamak için en ideal ayarları (hiperparametreleri) otomatik olarak bulma.

---

# Kodun Adım Adım Açıklaması

Bu proje, Python programlama dilinde `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost` ve `optuna` gibi popüler kütüphaneler kullanılarak geliştirilmiştir.

* *`pandas`*: Veri manipülasyonu ve analizi için temel kütüphanedir. CSV dosyalarını okumak, veri çerçeveleri oluşturmak ve özellikleri işlemek gibi tüm veri hazırlık adımlarında kullanıldı.
* *`numpy`*: Sayısal işlemler ve çok boyutlu diziler (array'ler) için kullanılır. Özellikle matematiksel hesaplamalarda ve model çıktılarının işlenmesinde verimli bir araçtır.
* *`scikit-learn`*: Makine öğrenimi algoritmaları ve yardımcı araçları için standart bir kütüphanedir. Projede **`LinearRegression`** gibi modeller, **`KFold`** gibi çapraz doğrulama teknikleri ve **`cross_val_predict`** gibi model değerlendirme fonksiyonları buradan gelmektedir.
* *`lightgbm`*, *`xgboost`*, *`catboost`*: Bu üç kütüphane, **"Gradient Boosting"** adı verilen bir makine öğrenimi algoritması ailesine aittir. 
* *`optuna`*: Modelin performansını en üst düzeye çıkarmak için en iyi parametreleri (örneğin, ağaç sayısı, öğrenme oranı) otomatik olarak bulan bir hiperparametre optimizasyon kütüphanesidir. Manuel deneme-yanılma yöntemine göre çok daha sistematik ve etkilidir.

# 1. Veri Yükleme ve Ön İşleme

Projenin ilk adımı, `train.csv` ve `test.csv` dosyalarının `pandas` kütüphanesi ile belleğe yüklenmesidir. Bu aşamada, `event_time` sütunu gibi zaman bilgileri doğru veri tipine dönüştürülerek sonraki işlemler için hazır hale getirilmiştir.

# 2. Gelişmiş Özellik Mühendisliği (`create_session_features` Fonksiyonu)

Modelin başarısı için en kritik adımlardan biri, ham veriden anlamlı özellikler çıkarmaktır. Bu fonksiyon, her kullanıcı seansı için aşağıdaki özellikleri oluşturur:

* *Temel Seans Özellikleri:* Toplam etkinlik sayısı (`total_events`), benzersiz ürün ve kategori sayıları (`unique_products`, `unique_categories`), seans süresi (`session_duration_sec`) gibi temel bilgiler.
* *Kullanıcı Davranışı Özellikleri:* Her seanstaki `view`, `add_to_cart` ve `buy` gibi etkinlik sayılarının yanı sıra, dönüşüm oranları (`view_to_cart_rate`, `cart_to_buy_rate`) hesaplanmıştır.
* *Kullanıcı Seviyesi Özellikler:* Bir kullanıcının tüm seansları boyunca ortalama etkinlik sayısı (`user_avg_events`) ve her bir etkinlik tipinin toplam sayısı (`user_view_count`, `user_add_to_cart_count`, `user_buy_count`) gibi kullanıcıya özel bilgiler eklenmiştir.
* *Lag (Geçmiş) Özelliği:* Profesyonel veri bilimi projelerinde sıkça kullanılan bu teknikte, her seans için aynı kullanıcının **bir önceki seansının değeri (`prev_session_value`)** modele eklenmiştir. Bu özellik, modelin kullanıcıların zaman içindeki davranış trendlerini öğrenmesini sağlayarak tahmin gücünü önemli ölçüde artırmıştır.

# 3. Model Optimizasyonu ve Ensemble (Stacking)

Bu aşama, projenin kalbidir ve en yüksek performansı elde etmek için stratejik bir yaklaşımla oluşturulmuştur.

1) **Otomatik Optimizasyon (`Optuna`)**: `Optuna` kütüphanesi kullanılarak, `LightGBM`, `XGBoost` ve `CatBoost` modelleri için en iyi parametre kombinasyonları (örneğin, ağaç sayısı, öğrenme oranı) otomatik olarak bulunmuştur. Bu, manuel deneme-yanılma yöntemine göre çok daha etkili ve zaman kazandıran bir yaklaşımdır.
   
* **Ne İçin Kullanıldı?** Bu modellerin performansı, hiperparametre ayarlarına bağlıdır. En iyi kombinasyonu bulmak, modelin potansiyelini maksimize etmek için hayati öneme sahiptir.
* **Neden Kullanıldı?** `Optuna`, deneme-yanılma sürecini otomatikleştirerek ve sonuçlara göre akıllıca yeni denemeler yaparak en iyi parametreleri çok daha kısa sürede bulur. Bu, projenin zaman verimliliğini ve nihai modelin doğruluğunu artırdı.


2) **Ensemble (Birleştirme) ve Stacking (Katmanlama)**: Bu projede **`LightGBM`, `XGBoost` ve `CatBoost`** olmak üzere üç farklı güçlü model kullanıldı. Her bir modelin tahminleri, tek bir modelin potansiyel hatalarını veya zayıf noktalarını telafi etmek için birleştirilmiştir.

* **Ne İçin Kullanıldı?** Üç farklı modelin tahminlerini birleştirerek daha kararlı ve doğru bir nihai tahmin elde etmek için kullanılmıştır. Stacking ise, modellerin tahminlerini basitçe ortalamak yerine, her bir modele ne kadar ağırlık vermesi gerektiğini öğrenen gelişmiş bir ensemble tekniğidir.
    
* **Neden Kullanıldı?** Ensemble, tahminin varyansını azaltarak modelin daha kararlı hale gelmesini sağlar. Stacking, her bir temel modelin güçlü yönlerini daha etkili bir şekilde birleştirerek nihai tahminin doğruluğunu artırır. Bu ileri seviye teknik, yarışmada sizi rakiplerinizin önüne geçiren en önemli faktörlerden biridir.
   
* **Nasıl Çalışır?**
  
        1.  **Out-of-Fold (OOF) Tahminleri:** Her bir temel model, çapraz doğrulama (`KFold`) kullanılarak eğitilir ve kendi eğitim verisi üzerinde tahminler oluşturur. Bu sayede her bir temel modelin, daha önce görmediği veriler üzerinde nasıl davrandığını öğrenmesi sağlanır.
        2.  **Meta-Model Eğitimi:** Oluşturulan OOF tahminleri (`lgb_oof`, `xgb_oof`, `cat_oof`), yeni bir veri seti gibi kabul edilir. Bu yeni veri seti, **`LinearRegression`** meta-modelini eğitmek için kullanılır. Meta-model, üç temel modelin tahminlerinden yola çıkarak nihai bir tahmin oluşturmayı öğrenir.

#### 4. Sonuç Dosyasının Oluşturulması

Elde edilen nihai tahminler, yarışma formatına uygun olarak bir CSV dosyasına (`submission_final_advanced.csv`) kaydedilmiş ve gönderim için hazır hale getirilmiştir.
Bu projede kullanılan teknikler, sadece bir yarışma kazanmaya yönelik değil, aynı zamanda gerçek dünya veri bilimi problemlerini çözmek için de geçerli ve etkili yöntemlerdir.



# ENG

### Final Version of the README Text

This text will clearly demonstrate the project's purpose, the methods used, and why each technique was so effective to potential employers and other data scientists. It has been prepared for the guncelsonkod.py file.

# Data Science Competition Project: User Session Value Prediction

This project was developed for a data science competition aimed at predicting the value of user sessions on an e-commerce platform. The achieved **MSE (Mean Squared Error) of 540** demonstrates the model's high accuracy in making predictions.

---

# Approach and Techniques Used

To achieve a high score, instead of a simple model, the following advanced data science techniques, used at a professional level, were combined:

* **Advanced Feature Engineering:** Creating strong signals based on data to enable the model to learn better.
* **Triple Ensemble and Stacking:** Combining the predictions of multiple powerful models to achieve a more stable and accurate final prediction.
* **Automated Hyperparameter Optimization:** Automatically finding the most ideal settings (hyperparameters) to capture the best performance of the models.

---

# Step-by-Step Explanation of the Code

This project was developed using popular libraries in the Python programming language such as `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, and `optuna`.

* *`pandas`*: It is the fundamental library for data manipulation and analysis. It was used in all data preparation steps such as reading CSV files, creating data frames, and processing features.
* *`numpy`*: It is used for numerical operations and multi-dimensional arrays. It is an efficient tool, especially in mathematical calculations and processing model outputs.
* *`scikit-learn`*: It is a standard library for machine learning algorithms and helper tools. Models like **`LinearRegression`**, cross-validation techniques like **`KFold`**, and model evaluation functions like **`cross_val_predict`** come from here.
* *`lightgbm`*, *`xgboost`*, *`catboost`*: These three libraries belong to a family of machine learning algorithms called **"Gradient Boosting."** They are the most preferred models in data science competitions due to their speed, performance, and ability to process large datasets.
* *`optuna`*: It is a hyperparameter optimization library that automatically finds the best parameters (e.g., number of trees, learning rate) to maximize model performance. It is much more systematic and effective than the manual trial-and-error method.

# 1. Data Loading and Preprocessing

The first step of the project is to load the `train.csv` and `test.csv` files into memory with the `pandas` library. At this stage, time information such as the `event_time` column was converted to the correct data type to be ready for subsequent operations.

# 2. Advanced Feature Engineering (`create_session_features` Function)

One of the most critical steps for the model's success is to extract meaningful features from the raw data. This function creates the following features for each user session:

* *Basic Session Features:* Basic information such as the total number of events (`total_events`), unique product and category counts (`unique_products`, `unique_categories`), and session duration (`session_duration_sec`).
* *User Behavior Features:* In addition to the number of events such as `view`, `add_to_cart`, and `buy` in each session, conversion rates (`view_to_cart_rate`, `cart_to_buy_rate`) were calculated.
* *User-Level Features:* User-specific information such as the average number of events per session for a user (`user_avg_events`) and the total count of each event type (`user_view_count`, `user_add_to_cart_count`, `user_buy_count`) were added.
* *Lag Feature:* In this technique, often used in professional data science projects, the **value of the previous session (`prev_session_value`)** for the same user in each session was added to the model. This feature significantly increased the model's predictive power by enabling it to learn user behavior trends over time.

# 3. Model Optimization and Ensemble (Stacking)

This stage is the heart of the project and was created with a strategic approach to achieve the highest performance.

1) **Automated Optimization (`Optuna`)**: Using the `Optuna` library, the best parameter combinations (e.g., number of trees, learning rate) for `LightGBM`, `XGBoost`, and `CatBoost` models were automatically found. This is a much more effective and time-saving approach than the manual trial-and-error method.

   * **What Was It Used For?** The performance of these models depends on hyperparameter settings. Finding the best combination is vital to maximize the model's potential.
   * **Why Was It Used?** `Optuna` automates the trial-and-error process and finds the best parameters in a much shorter time by intelligently making new trials based on the results. This increased the project's time efficiency and the final model's accuracy.

2) **Ensemble (Combining) and Stacking (Layering)**: In this project, three different powerful models were used: **`LightGBM`**, **`XGBoost`**, and **`CatBoost`**. The predictions of each model were combined to compensate for the potential errors or weaknesses of a single model.

   * **What Was It Used For?** It was used to get a more stable and accurate final prediction by combining the predictions of three different models. Stacking, on the other hand, is an advanced ensemble technique that learns how much weight to give to each model, instead of simply averaging their predictions.
   * **Why Was It Used?** Ensemble reduces the variance of the prediction, making the model more stable. Stacking increases the accuracy of the final prediction by combining the strengths of each base model more effectively. This advanced technique is one of the most important factors that put you ahead of your competitors in the competition.
   * **How Does It Work?**
     1. **Out-of-Fold (OOF) Predictions:** This is the most important step of stacking. Each base model is trained using cross-validation (`KFold`) and makes **predictions on its own training data**. This enables each base model to learn how it behaves on data it has not seen before.
     2. **Meta-Model Training:** The created OOF predictions (`lgb_oof`, `xgb_oof`, `cat_oof`) are treated as a new dataset. This new dataset is used to train the **`LinearRegression`** meta-model. The meta-model learns to create a final prediction based on the predictions of the three base models.

# 4. Creating the Submission File

The final predictions obtained were saved to a CSV file (`submission_final_advanced.csv`) in the format required by the competition and made ready for submission.

The techniques used in this project are not only for winning a competition but are also valid and effective methods for solving real-world data science problems.
