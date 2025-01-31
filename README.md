# DETEKSI KELAINAN PARU PADA CHEST CT SCAN MENGGUNAKAN PYTORCH

## PENGOLAHAN CITRA DIGITAL

### Dosen Pengampu:
Leni Fitriani, ST., M.Kom

### Disusun oleh:
- **Muhammad Hanif** (2206094)
- **Rina Rismawati** (2206076)

**PROGRAM STUDI TEKNIK INFORMATIKA**  
**INSTITUT TEKNOLOGI GARUT**  
**2025**  

---

## KATA PENGANTAR
Alhamdulillah, segala puji penyusun panjatkan kehadirat Allah SWT atas rahmat dan hidayah-Nya, sehingga laporan ini dapat terselesaikan. Laporan ini disusun untuk memenuhi UAS mata kuliah Pengolahan Citra Digital.

---

## DAFTAR ISI
1. [Pendahuluan](#pendahuluan)
2. [Metode Penelitian](#metode-penelitian)
3. [Pembahasan](#pembahasan)
4. [Kesimpulan](#kesimpulan)
5. [Daftar Pustaka](#daftar-pustaka)

---

## 1. Pendahuluan

### 1.1 Latar Belakang
Kemajuan teknologi AI, khususnya deep learning, memberikan dampak besar dalam bidang kesehatan. Deteksi kelainan paru akibat COVID-19 dapat dilakukan secara otomatis menggunakan model berbasis deep learning seperti ResNet-18.

### 1.2 Penelitian atau Teori Terkait
Beberapa penelitian terdahulu menunjukkan bahwa metode deep learning pada Chest CT Scan dapat meningkatkan akurasi deteksi COVID-19 dibandingkan metode konvensional.

### 1.3 Tujuan Penelitian
- Mengembangkan model deep learning berbasis PyTorch untuk deteksi kelainan paru.
- Meningkatkan akurasi diagnosis dibandingkan metode manual.

---

## 2. Metode Penelitian

### 2.1 Langkah-langkah Penelitian
1. **Pengumpulan Dataset** dari Kaggle.
2. **Preprocessing Data**: Normalisasi, augmentasi, dan segmentasi.
3. **Pengembangan Model** menggunakan CNN berbasis ResNet-18.
4. **Evaluasi Model** dengan akurasi, sensitivitas, dan spesifisitas.
5. **Implementasi dan Validasi** dengan data real-world.

### 2.2 Visualisasi Model
Proses penelitian dimulai dengan pengumpulan data, diikuti preprocessing, pelatihan model, dan klasifikasi kelainan paru menggunakan PyTorch.

---

## 3. Pembahasan

### 3.1 Bahan Penelitian
Dataset berisi gambar Chest CT Scan dengan kondisi normal dan terinfeksi COVID-19.

### 3.2 Akuisisi Citra
Dataset diperoleh dari repositori publik dan diproses melalui tahap normalisasi dan augmentasi.

### 3.3 Pre-Processing
- **Resizing** untuk ukuran seragam.
- **Normalisasi** piksel dalam skala 0-1.
- **Augmentasi Data** seperti rotasi dan flipping.

### 3.4 Perancangan Sistem
Model dibuat menggunakan PyTorch dengan arsitektur **ResNet-18**, serta dioptimalkan dengan Adam optimizer dan cross-entropy loss.

### 3.5 Hasil
Model dilatih selama 15 epoch dengan akurasi akhir **82.22%**.

### 3.6 Accuracy
Model memiliki performa cukup baik tetapi masih perlu dioptimalkan untuk mendeteksi kelainan paru lebih akurat.

---

## 4. Kesimpulan

### 4.1 Ringkasan Temuan
- Model ResNet-18 dapat mendeteksi kelainan paru dengan akurasi tinggi.
- Preprocessing berkontribusi dalam peningkatan akurasi.
- Model memiliki potensi untuk membantu diagnosis medis.

### 4.2 Batasan Pekerjaan
- Dataset terbatas pada satu repositori.
- Model mengalami **overfitting** pada beberapa epoch.

### 4.3 Rekomendasi
- Menggunakan dataset lebih beragam.
- Menerapkan **regularisasi** untuk mengurangi overfitting.
- Menguji model dengan arsitektur lain seperti EfficientNet.

---

## 5. Daftar Pustaka
- Asefi, H., & Safaie, A. (2020). *The Role of Chest CT Scan in Diagnosis of COVID-19*. 
- Fukuhara, T. et al. (2019). *Usefulness of chest CT scan for head and neck cancer*.
- Pecoraro, M. et al. (2021). *Cross-sectional analysis of follow-up chest MRI and chest CT scans*.
- Poortahmasebi, V. et al. (2020). *Clinical Performance of RT-PCR and Chest CT Scan for Covid-19 Diagnosis*.
- Tenda, E. D. et al. (2020). *The Importance of Chest CT Scan in COVID-19*.
