![image](https://github.com/user-attachments/assets/bf07211e-25d5-48ed-93c1-51fdadd5d1eb)# PENGOLAHAN CITRA DIGITAL  
## DETEKSI KELAINAN PARU PADA CHEST CT SCAN MENGGUNAKAN PYTORCH  

Dibuat untuk mengerjakan Ulangan Akhir Semester (UAS) mata kuliah Pengolahan Citra Digital  
**Dosen Pengampu:** Leni Fitriani, ST., M.Kom  

---

### Disusun oleh:  
**Muhammad Hanif** (2206094)  
**Rina Rismawati** (2206076)  

---

### PROGRAM STUDI TEKNIK INFORMATIKA  
### INSTITUT TEKNOLOGI GARUT  
### 2025  

---

## KATA PENGANTAR  

Alhamdulillah, segala puji penyusun panjatkan kehadirat Allah SWT atas rahmat dan hidayah-Nya penyusun dapat menyelesaikan laporan Ujian Akhir Semester (UAS) mata kuliah Pengolahan Citra Digital dengan judul *Deteksi Kelainan Paru Pada Chest CT Scan Menggunakan Pytorch*.  

Adapun tujuan penyusunan laporan praktikum ini adalah untuk memenuhi UAS mata kuliah Pengolahan Citra Digital.  
Penyusun menyadari bahwa laporan ini jauh dari kata sempurna baik dalam kata ataupun teknik penyusunan. Oleh karena itu, penyusun mengharapkan kritik dan saran yang membangun dari pembaca.  

Akhir kata, penyusun mengucapkan mohon maaf kepada semua pihak apabila dalam penyusunan laporan ini terdapat kesalahan.  

**Januari, 2025**  

**Penyusun**  

---

## DAFTAR GAMBAR  

- **Gambar 2.1** Langkah Penelitian - *halaman 5*  
- **Gambar 3.1** Hasil Training - *halaman 9*  
- **Gambar 3.2** Positive for Cancer - *halaman 11*  

---

## DAFTAR ISI  

- **KATA PENGANTAR** i  
- **DAFTAR GAMBAR** ii  
- **DAFTAR ISI** iii  

### **BAB I PENDAHULUAN** 1  
- **1.1.** Latar Belakang 1  
- **1.2.** Penelitian atau Teori Terkait 2  
- **1.3.** Tujuan Penelitian 4  

### **BAB II METODE PENELITIAN** 5  
- **2.1.** Langkah-langkah Penelitian 5  
- **2.2.** Visualisasi Model 6  

### **BAB III PEMBAHASAN** 7  
- **3.1.** Bahan Penelitian 7  
- **3.2.** Akuisisi Citra 7  
- **3.3.** Pre-Processing 8  
- **3.4.** Perancangan Sistem 8  
- **3.5.** Hasil 9  
- **3.6.** Accuracy 10  

### **BAB IV KESIMPULAN** 12  
- **4.1.** Ringkasan Temuan 12  
- **4.2.** Batasan Pekerjaan 12  
- **4.3.** Rekomendasi untuk Pekerjaan di Masa Depan 13  

### **DAFTAR PUSTAKA** 14  


# BAB I  
## PENDAHULUAN  

### 1.1. Latar Belakang  
Kemajuan teknologi dalam bidang kecerdasan buatan (*Artificial Intelligence*/AI) telah memberikan dampak yang signifikan dalam berbagai sektor, termasuk di bidang kesehatan. Dengan adanya AI, banyak proses yang sebelumnya bergantung pada tenaga manusia kini dapat diotomatisasi dengan lebih cepat dan akurat. Salah satu penerapan AI yang berkembang pesat adalah dalam bidang pencitraan medis, yang memungkinkan diagnosis penyakit secara lebih efektif. *Deep learning*, sebagai bagian dari AI, telah memainkan peran penting dalam pengolahan citra medis dengan memberikan hasil yang lebih akurat dibandingkan metode tradisional.  

Pandemi COVID-19 yang melanda dunia sejak akhir 2019 telah menjadi tantangan besar bagi sistem kesehatan global. COVID-19 merupakan penyakit infeksi pernapasan yang disebabkan oleh virus SARS-CoV-2 dan dapat menyebabkan pneumonia berat serta komplikasi lainnya pada paru-paru. Salah satu metode utama untuk mendiagnosis infeksi ini adalah melalui teknik pencitraan medis seperti *Chest CT Scan*, yang memberikan gambaran detail kondisi paru-paru pasien. Namun, dalam situasi pandemi dengan jumlah pasien yang tinggi, analisis manual citra medis oleh tenaga medis menjadi sangat terbatas dan rawan kesalahan. Oleh karena itu, diperlukan sistem berbasis *deep learning* yang mampu mendeteksi COVID-19 secara otomatis dengan akurasi tinggi.  

Dengan perkembangan teknologi *deep learning*, kini model berbasis jaringan saraf tiruan (*Neural Networks*) dapat digunakan untuk mendeteksi kelainan paru akibat COVID-19 secara otomatis. Salah satu arsitektur jaringan yang sering digunakan adalah *ResNet-18*, bagian dari keluarga *Residual Network* (ResNet), yang terkenal karena kemampuannya dalam menangani permasalahan *deep learning* seperti *vanishing gradient*. Model ini memungkinkan sistem untuk belajar fitur kompleks dari gambar *CT Scan* paru-paru dan mengklasifikasikannya dengan lebih efektif. Implementasi *deep learning* dalam pencitraan medis tidak hanya meningkatkan efisiensi diagnosis tetapi juga membantu mengurangi beban kerja tenaga medis, terutama dalam situasi darurat seperti pandemi.  

### 1.2. Penelitian atau Teori Terkait  
Sejumlah penelitian telah dilakukan dalam penerapan *deep learning* untuk mendeteksi COVID-19 dari pencitraan medis. Diantaranya:  

1. **"The Role of Chest CT Scan in Diagnosis of COVID-19"** (Asefi & Safaie, 2020) menunjukkan bahwa penggunaan *CT scan* dada dalam mendeteksi COVID-19 pneumonia memiliki sensitivitas yang tinggi, dengan angka mencapai 97% pada pasien dengan hasil RT-PCR positif. Mereka juga menemukan bahwa lebih dari 70% pasien yang awalnya memiliki hasil RT-PCR negatif masih menunjukkan temuan *CT* yang khas untuk COVID-19, yang menunjukkan keterbatasan dari metode RT-PCR dalam fase awal infeksi.  

2. **"Clinical characteristics of 138 hospitalized patients with 2019 novel coronavirus–infected pneumonia in Wuhan, China"** (Poortahmasebi et al., 2020) menyelidiki karakteristik klinis dari 138 pasien yang terinfeksi pneumonia akibat virus corona baru di Wuhan, Tiongkok. Penelitian ini menyoroti pentingnya diagnosis dini dan tepat untuk penanganan COVID-19.  

3. **"The Importance of Chest CT Scan in COVID-19: A Case Series"** (Tenda et al., 2020) menunjukkan bahwa *CT scan* dada memiliki kemampuan yang signifikan dalam mendeteksi COVID-19, terutama pada pasien yang menunjukkan gejala moderat dan hasil rontgen dada yang tidak konklusif. Dalam studi ini, ditemukan bahwa pemeriksaan *CT* dapat mengidentifikasi temuan khas seperti *ground-glass opacities* dan konsolidasi paru-paru yang menunjukkan pneumonia.  

4. **"Cross sectional analysis of follow-up chest MRI and chest CT scans in patients previously affected by COVID-19"** (Pecoraro et al., 2021) menunjukkan bahwa *MRI* memiliki potensi yang besar dalam evaluasi penyakit paru interstitial dengan menunjukkan kemampuan untuk mendeteksi perubahan morfologis dan fungsional yang tidak terjangkau oleh *CT*.  

5. **"Usefulness of chest CT scan for head and neck cancer"** (Fukuhara et al., 2019) menunjukkan bahwa deteksi nodul paru kecil melalui *CT* memiliki dampak signifikan terhadap hasil klinis pasien, terutama dalam konteks diagnosis dan manajemen nodul tersebut.  

### 1.3. Tujuan Penelitian  
Penelitian ini bertujuan untuk mengembangkan model berbasis *deep learning* menggunakan PyTorch untuk:  

1. Mendeteksi dan mengklasifikasikan kelainan paru akibat COVID-19 pada gambar *Chest CT Scan*.  
2. Meningkatkan akurasi dan efisiensi dalam diagnosis COVID-19 dibandingkan metode konvensional.  

# BAB II  
## METODE PENELITIAN  

### 2.1. Langkah-langkah Penelitian  

![Gambar 2.1 Langkah Penelitian](image-path)  

Penelitian ini dilakukan menggunakan pendekatan Deep Learning berbasis PyTorch untuk mendeteksi kelainan paru pada gambar Chest CT Scan. Metode penelitian terdiri dari beberapa tahapan, yaitu:  

1. **Pengumpulan Dataset**  
   Dataset Chest CT Scan yang digunakan dalam penelitian ini diperoleh dari repositori publik, Kaggle, yang telah banyak digunakan dalam penelitian terdahulu (Poortahmasebi et al., 2021). Dataset ini mencakup gambar CT pasien dengan kondisi normal dan pasien yang terinfeksi COVID-19.  

2. **Preprocessing Data**  
   Tahap ini bertujuan untuk meningkatkan kualitas data sebelum diproses lebih lanjut. Langkah-langkah yang dilakukan meliputi:  
   - **Normalisasi Data**: Mengubah nilai piksel gambar CT ke skala 0-1 untuk mempermudah proses pelatihan model (Pecoraro et al., 2021).  
   - **Augmentasi Data**: Melakukan rotasi, flipping, atau cropping gambar untuk mengatasi masalah ketidakseimbangan data.  
   - **Segmentasi**: Menggunakan metode thresholding untuk memisahkan area paru dari gambar CT lainnya (Asefi & Safaie, 2021).  

3. **Pengembangan Model Deep Learning**  
   Penelitian ini menggunakan arsitektur Convolutional Neural Network (CNN) berbasis PyTorch, dengan beberapa langkah berikut:  
   - **Transfer Learning**: Menggunakan model pralatih (pre-trained model) seperti ResNet atau EfficientNet yang telah terbukti efektif dalam tugas klasifikasi citra (Fukuhara et al., 2021).  
   - **Pelatihan Model**: Model dilatih menggunakan algoritma Adam optimizer dengan fungsi kehilangan cross-entropy loss.  
   - **Parameter Tuning**: Melakukan pengaturan parameter seperti learning rate, jumlah epochs, dan ukuran batch untuk mencapai performa terbaik.  

4. **Evaluasi Model**  
   Kinerja model dievaluasi menggunakan beberapa metrik, yaitu:  
   - **Akurasi**: Persentase prediksi yang benar dari seluruh data uji.  
   - **Sensitivitas dan Spesifisitas**: Kemampuan model untuk mendeteksi kasus positif dan negatif dengan benar (Tenda et al., 2021).  
   - **F1-Score**: Pengukuran gabungan dari presisi dan sensitivitas untuk menghindari bias pada data tidak seimbang.  

5. **Implementasi dan Validasi**  
   Model yang telah dikembangkan diimplementasikan dan divalidasi menggunakan data real-world dari rumah sakit atau dataset baru yang belum pernah digunakan saat pelatihan. Proses ini bertujuan untuk menguji performa model dalam kondisi nyata (Asefi & Safaie, 2021).  

### 2.2. Visualisasi Model  

Metode yang digunakan dalam penelitian ini terdiri dari beberapa tahap utama yang saling berhubungan. Proses dimulai dengan pengumpulan data berupa gambar Chest CT Scan, yang kemudian diproses melalui pre-processing untuk meningkatkan kualitas dan keseragaman data. Selanjutnya, model deep learning berbasis PyTorch dilatih menggunakan dataset yang telah dipersiapkan. Setelah model selesai dilatih, dilakukan klasifikasi untuk mendeteksi dan mengidentifikasi kelainan paru, khususnya COVID-19. Terakhir, hasil prediksi dievaluasi menggunakan berbagai metrik untuk memastikan akurasi dan kinerja model dalam diagnosis otomatis berbasis pencitraan medis.  

# BAB III  
## PEMBAHASAN  

### 3.1. Bahan Penelitian  
Penelitian ini menggunakan dataset gambar Chest CT Scan yang berisi citra paru-paru dengan berbagai kondisi, termasuk yang terinfeksi COVID-19 dan yang normal. Dataset ini diperoleh dari sumber terbuka yang telah dikurasi untuk memastikan kualitas dan relevansinya dalam mendeteksi kelainan paru.  
Setiap gambar dalam dataset mengalami tahap pre-processing, seperti normalisasi, augmentasi data, dan resizing, untuk meningkatkan kualitas dan keseragaman data yang digunakan dalam pelatihan model. Model deep learning yang digunakan dalam penelitian ini berbasis PyTorch, dengan arsitektur jaringan saraf tiruan seperti ResNet-18, yang mampu mengenali pola dalam gambar dengan tingkat akurasi yang tinggi.  
Selain itu, penelitian ini memanfaatkan perangkat keras dengan GPU untuk mempercepat proses training model. Evaluasi kinerja model dilakukan menggunakan metrik seperti akurasi, precision, recall, dan F1-score, yang memberikan gambaran tentang efektivitas model dalam mendeteksi COVID-19 dari citra Chest CT Scan.  

### 3.2. Akuisisi Citra  
Di tahap ini, peneliti melakukan akuisisi citra dengan mengumpulkan dan menyiapkan gambar chest CT scan yang digunakan sebagai data utama dalam model deteksi COVID-19. Sumber citra diperoleh dari dataset yang tersedia secara publik, yang berisi gambar paru-paru dalam berbagai kondisi, termasuk paru-paru normal, terinfeksi COVID-19, dan dengan kelainan lainnya.  
Setelah citra dikumpulkan, dilakukan tahap pre-processing untuk memastikan kualitas gambar sesuai dengan kebutuhan model deep learning. Proses ini mencakup resizing gambar agar memiliki ukuran yang seragam, normalisasi intensitas piksel untuk meningkatkan kontras, serta augmentasi data guna memperluas variasi gambar dan mencegah overfitting saat pelatihan model.  
Dengan akuisisi citra yang sistematis dan pre-processing yang tepat, dataset yang digunakan dalam penelitian ini dapat memberikan informasi yang lebih representatif dan mendukung peningkatan akurasi model dalam mendeteksi COVID-19 melalui chest CT scan.  

### 3.3. Pre-Processing  
Pada tahap ini, peneliti melakukan preprocessing data untuk memastikan bahwa citra yang digunakan dalam pelatihan model memiliki kualitas yang optimal. Proses preprocessing mencakup beberapa langkah utama, yaitu resizing, normalisasi, dan augmentasi data.  
- **Resizing** dilakukan untuk menyesuaikan ukuran gambar agar seragam, sehingga dapat diproses oleh model deep learning tanpa adanya perbedaan dimensi yang signifikan.  
- **Normalisasi** diterapkan dengan merubah nilai piksel ke dalam rentang tertentu, biasanya antara 0 hingga 1, guna meningkatkan stabilitas pelatihan dan mempercepat konvergensi model.  
- **Augmentasi data** digunakan untuk memperbanyak variasi citra dengan teknik seperti rotasi, flipping, dan perubahan kontras. Hal ini bertujuan untuk meningkatkan kemampuan model dalam mengenali pola pada berbagai kondisi pencitraan.  

Dengan preprocessing yang baik, data yang digunakan menjadi lebih berkualitas dan membantu model dalam menghasilkan prediksi yang lebih akurat.  

### 3.4. Perancangan Sistem  
Pada tahap ini, peneliti melakukan pembuatan sistem deteksi kelainan paru menggunakan deep learning berbasis PyTorch. Sistem ini dikembangkan dalam Google Colab untuk memanfaatkan komputasi berbasis cloud yang lebih efisien. Langkah-langkah utama dalam pembuatan program meliputi pemuatan dataset, preprocessing data, pelatihan model, evaluasi performa, serta prediksi terhadap citra uji.  
Pertama, dataset yang berisi gambar Chest CT Scan dimuat ke dalam sistem. Data ini kemudian melalui tahap preprocessing yang mencakup resizing, normalisasi, dan augmentasi untuk meningkatkan kualitas input. Setelah data siap, model deep learning dikembangkan menggunakan arsitektur jaringan saraf, di mana dalam penelitian ini digunakan model ResNet-18 yang telah terbukti efektif dalam klasifikasi gambar medis.  
Model kemudian dilatih menggunakan data latih dengan optimasi parameter melalui algoritma seperti Adam dan fungsi loss yang sesuai untuk tugas klasifikasi. Setelah pelatihan selesai, model dievaluasi menggunakan data uji untuk mengukur tingkat akurasi dan performa lainnya. Terakhir, sistem digunakan untuk melakukan prediksi terhadap citra baru guna mengklasifikasikan apakah gambar mengandung kelainan paru atau tidak. Dengan pendekatan ini, diharapkan sistem dapat membantu dalam mendeteksi kelainan paru dengan akurasi yang lebih tinggi dibandingkan metode konvensional.  

### 3.5. Hasil  

![Gambar 3.1 Hasil Training](image-path)  

Hasil pelatihan model dengan menganalisis nilai loss dan akurasi pada dataset pelatihan dan pengujian. Gambar di atas menunjukkan hasil pelatihan model selama 15 epoch, di mana setiap epoch merepresentasikan satu siklus penuh pelatihan menggunakan seluruh data latih.  
Dari hasil pelatihan, terlihat bahwa train loss mengalami penurunan secara signifikan, dari 1.0259 pada epoch pertama menjadi sekitar 0.1679 pada epoch terakhir. Hal ini menunjukkan bahwa model berhasil mempelajari pola dari data latih dengan baik. Akurasi pada data latih juga meningkat dari 0.5494 pada epoch pertama menjadi sekitar 0.9447 pada epoch terakhir, yang mengindikasikan bahwa model mampu mengklasifikasikan data latih dengan baik.  
Untuk hasil pengujian, test loss berfluktuasi dan sempat meningkat pada beberapa epoch awal, yang dapat mengindikasikan adanya overfitting. Namun, test accuracy secara keseluruhan mengalami peningkatan, dari 0.2994 pada epoch pertama menjadi 0.8227 pada epoch terakhir. Nilai ini menunjukkan bahwa model memiliki performa yang cukup baik dalam mengklasifikasikan data uji.  
Total waktu yang dibutuhkan untuk melatih model adalah sekitar 3485.882 detik. Secara keseluruhan, hasil pelatihan menunjukkan bahwa model deep learning berbasis ResNet-18 dapat mendeteksi kelainan paru dengan akurasi yang cukup tinggi, meskipun masih terdapat ruang untuk peningkatan lebih lanjut.  

### 3.6. Accuracy  
Berdasarkan eksperimen yang telah dilakukan, didapatkan hasil sebagai berikut:  
- Akurasi model yang diuji pada dataset validasi mencapai 82.22%.  
- Penggunaan transfer learning dengan ResNet-18 masih memiliki keterbatasan dalam mendeteksi kelainan paru dengan akurasi tinggi.  
- Dengan evaluasi menggunakan confusion matrix dan visualisasi hasil prediksi, dapat disimpulkan bahwa model masih perlu dioptimalkan untuk meningkatkan performa dalam mendeteksi kelainan paru akibat COVID-19.  


![Gambar 3.2 Positive for Cancer](image-path)


# BAB IV  
## KESIMPULAN  

### 4.1. Ringkasan Temuan  
1. Model deep learning berbasis ResNet-18 mampu mendeteksi kelainan paru akibat COVID-19 dari gambar Chest CT Scan dengan akurasi yang cukup tinggi.  
2. Preprocessing data seperti normalisasi, augmentasi, dan segmentasi berperan penting dalam meningkatkan kualitas data dan akurasi model.  
3. Hasil pelatihan menunjukkan peningkatan akurasi dari 54,94% menjadi 94,47% pada data latih dan 82,22% pada data uji, meskipun terdapat indikasi overfitting.  
4. Penggunaan PyTorch dan Google Colab memungkinkan pelatihan model dengan efisiensi tinggi menggunakan komputasi berbasis cloud.  
5. Model yang dikembangkan memiliki potensi untuk membantu tenaga medis dalam diagnosis COVID-19, terutama dalam kondisi darurat dengan jumlah pasien tinggi.  

### 4.2. Batasan Pekerjaan  
1. Dataset yang digunakan masih terbatas, hanya berasal dari satu sumber repositori publik (Kaggle) tanpa validasi dari sumber rumah sakit yang berbeda.  
2. Model cenderung mengalami overfitting, yang ditunjukkan oleh fluktuasi test loss pada beberapa epoch.  
3. Tidak dilakukan perbandingan dengan metode deep learning lain seperti EfficientNet atau DenseNet, yang mungkin memberikan hasil lebih optimal.  
4. Evaluasi hanya menggunakan metrik akurasi, sensitivitas, spesifisitas, dan F1-score, tanpa analisis lebih lanjut mengenai interpretabilitas model.  
5. Implementasi di dunia nyata belum diuji dengan data real-time dari rumah sakit atau institusi medis lainnya.  

### 4.3. Rekomendasi untuk Pekerjaan di Masa Depan  
1. Menggunakan dataset yang lebih besar dan beragam dari berbagai institusi medis untuk meningkatkan generalisasi model.  
2. Menerapkan teknik regularisasi dan fine-tuning hyperparameter untuk mengurangi overfitting.  
3. Menguji model dengan arsitektur deep learning lain seperti EfficientNet, DenseNet, atau Vision Transformers untuk meningkatkan performa.  
4. Melakukan validasi eksternal dengan data dari sumber medis yang kredibel guna memastikan akurasi model dalam kondisi nyata.  
5. Mengembangkan sistem berbasis web atau aplikasi yang dapat digunakan oleh tenaga medis untuk mendeteksi kelainan paru secara langsung.


# DAFTAR PUSTAKA  

1. Asefi, H., & Safaie, A. (2020). The Role of Chest CT Scan in Diagnosis of COVID-19. *4(11)*, 1–5. https://doi.org/10.22114/ajem.v4i2s.451  
2. Fukuhara, T., Fujiwara, K., Fujii, T., & Takeda, K. (2019). Usefulness of chest CT scan for head and neck cancer. *Auris Nasus Larynx, 42(1)*, 49–52. https://doi.org/10.1016/j.anl.2014.08.013  
3. Maesaroh, S., Afiyati, Hakim, L., Sari, Y. S., Yusuf, M., Perkasa, E. B., Utami, W. S., Saptadi, N. T. S., Mutmainah, S., Khairunnas, Harahap, E. P., Alamin, Z., Karima, I. S., Saputra, A., & Mubarak, R. (2024). *BAHASA PEMROGRAMAN PYTHON* (C. E. Muhamad Rizal Kurnia, M.E. (ed.)). PENERBIT PT SADA KURNIA PUSTAKA.  
4. Pecoraro, M., Cipollari, S., Marchitelli, L., Messina, E., Del, M., Nicola, M., Rosa, M., Marco, C., Carlo, F., & Valeria, C. (2021). Cross‑sectional analysis of follow‑up chest MRI and chest CT scans in patients previously affected by COVID‑19. *La Radiologia Medica, 126(10)*, 1273–1281. https://doi.org/10.1007/s11547-021-01390-4  
5. Poortahmasebi, V., Zandi, M., Soltani, S., & Jazayeri, S. M. (2020). Clinical Performance of RT-PCR and Chest CT Scan for Covid-19 Diagnosis; a Systematic Review. *ADVANCED JOURNAL OF EMERGENCY MEDICINE, 4*, 1–7. https://doi.org/10.22114/ajem.v4i2s.459  
6. Tenda, E. D., Yulianti, M., Asaf, M. M., Yunus, R. E., Septiyanti, W., Wulani, V., Pitoyo, C. W., Rumende, C. M., & Setiati, S. (2020). The Importance of Chest CT Scan in COVID-19: A Case Series. *Acta Medica Indonesiana, 52(1)*, 68–73.
