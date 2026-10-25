# Praktik Terpadu RET503 (Minggu 4-7): Dari COCO ke Detektor YOLO

- NIM / Nama: 4222401081 / Alicia Theresa Sianturi
- Dijalankan: 24-10-2026 16:30:52, di Google Colab
- Model yolo26n, 30 epoch, imgsz 416 (khusus E3: 640), batch 16, seed 0
- Sidik jari label dataset: A65541F45E9F
- Pengerjaan: satu notebook Colab yang memuat pengambilan data, pemeriksaan label, pelatihan E1-E3, dan evaluasi

## 1. Data (Tahap 1-2)

Data diambil dari COCO val2017 dengan 3 kelas: dog (0), cup (1), toothbrush (2). Kelas toothbrush sengaja dipilih karena jumlahnya sedikit di COCO.

| Item | Jumlah |
|---|---|
| Gambar berobjek | 200 |
| Gambar latar (tanpa objek) | 20 |
| Total gambar | 220 |
| Kotak dog / cup / toothbrush | 72 / 259 / 57 (total 388) |
| Train / val / test | 154 / 44 / 22 gambar (70/20/10) |
| Objek di val | 84 |
| Objek di test | 37 |

Pemeriksaan anotasi: label salah = 0, peringatan kotak sangat kecil (<8 px) = 13. Konversi YOLO ke COCO lalu balik ke YOLO: 220 gambar, 388 kotak, 3 kategori, galat maksimum 1.1e-16.

## 2. Hasil eksperimen (Tahap 3-5)

Ketiga eksperimen memakai data, epoch, batch, dan seed yang sama. Bedanya hanya kolom "Perubahan". Angka dari data val.

| Eksperimen | Perubahan | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| E1_pretrained | - | 0.537 | 0.341 | 0.672 | 0.516 |
| E2_scratch | - | 0.003 | 0.001 | 0.001 | 0.127 |
| E3_tuning | imgsz=640 | 0.657 | 0.473 | 0.869 | 0.567 |

Model terpilih (mAP50-95 val tertinggi): E3_tuning.
Hasil di data test (dipakai sekali saja): mAP50 0.630, mAP50-95 0.443.

Hasil per kelas di data val (E1 vs E3):

| Kelas | Objek di val | E1 mAP50-95 | E3 mAP50-95 | E1 recall | E3 recall |
|---|---|---|---|---|---|
| dog | 16 | 0.563 | 0.720 | 0.812 | 0.938 |
| cup | 52 | 0.385 | 0.516 | 0.673 | 0.654 |
| toothbrush | 16 | 0.073 | 0.184 | 0.062 | 0.109 |

## 3. Analisis

**Hal penting yang perlu dibaca dulu: best.pt E1 dan E3 berasal dari epoch 1.** Pada E1 dan E3, nilai val tertinggi (dipakai Ultralytics untuk memilih best.pt) justru muncul di epoch pertama: mAP50-95 0.342 untuk E1 dan 0.471 untuk E3. Angka tabel di bagian 2 (0.341 dan 0.473) sama dengan angka epoch 1 itu, jadi model yang dipilih dan diuji di data test pada dasarnya bobot setelah satu epoch. Setelah itu mAP val turun sampai titik terendah di epoch 9 (E1, 0.120) dan epoch 12 (E3, 0.133), lalu naik lagi ke sekitar 0.24 di epoch 30 tanpa pernah melewati nilai epoch 1.

| Eksperimen | mAP50-95 val epoch 1 | Terendah (epoch) | Epoch 30 | best.pt dari |
|---|---|---|---|---|
| E1_pretrained | 0.342 | 0.120 (9) | 0.239 | epoch 1 |
| E2_scratch | 0.000 | - | < 0.001 | sekitar epoch 22 |
| E3_tuning | 0.471 | 0.133 (12) | 0.228 | epoch 1 |

Penyebabnya belum diuji. Dugaan: (a) dog, cup, dan toothbrush termasuk 80 kelas COCO, jadi model pretrained sudah mengenal ketiganya dan satu epoch cukup untuk menyesuaikan bagian keluaran; (b) pada epoch-epoch awal learning rate masih tinggi dan model sedang warmup (warmup_epochs 3, optimizer auto), sehingga bobot bagus tadi terganggu dulu sebelum pulih; (c) val hanya 44 gambar, jadi nilai antar-epoch memang bergoyang. Akibatnya, kurva results.png tidak menggambarkan model terpilih: kurva menunjukkan epoch 2-30, sedangkan model terpilih adalah epoch 1.

**E1 vs E2.** Perbedaannya sangat besar: mAP50-95 0.341 untuk E1 dan 0.001 untuk E2, dan mAP E2 tidak pernah melewati 0.001 selama 30 epoch. Loss E2 juga hampir tidak turun: cls_loss train 6.54 menjadi 5.63, dibanding E1 dari 3.31 menjadi 1.06. Dugaan: 154 gambar latih terlalu sedikit untuk model dengan bobot acak, sedangkan E1 membawa pengetahuan tepi, tekstur, dan bentuk dari COCO. Ini sejalan dengan materi transfer learning, tetapi penyebabnya belum diuji terpisah.

**E1 vs E3.** Dengan imgsz dinaikkan dari 416 ke 640 (satu-satunya perubahan), mAP50-95 naik dari 0.341 ke 0.473, precision dari 0.672 ke 0.869, dan recall dari 0.516 ke 0.567. Kenaikan terlihat di ketiga kelas (dog 0.563 ke 0.720, cup 0.385 ke 0.516, toothbrush 0.073 ke 0.184). Perubahan dipertahankan, dan E3 menjadi model terpilih. Harganya: waktu inferensi naik dari 6.3 ms ke 9.3 ms, wajar karena jumlah piksel input sekitar 2,4 kali lebih banyak. Waktu latih hampir sama (1.8 dan 1.7 menit). Karena kedua angka berasal dari bobot epoch 1 dan dari satu seed, selisih ini mungkin sebagian kebetulan.

**Ketidakseimbangan kelas.** Jumlah kotak: cup 259, dog 72, toothbrush 57. Cup paling banyak, tetapi skornya di tengah; dog punya kotak lebih sedikit dari cup tetapi skornya tertinggi; toothbrush paling lemah (mAP50-95 0.073 di E1 dan 0.184 di E3, recall 0.062 dan 0.109) padahal jumlah kotaknya mendekati dog. Jadi jumlah kotak saja tidak menjelaskan hasil per kelas. Dugaan: sikat gigi berukuran kecil dan tipis sehingga sulit dideteksi, tetapi ini belum dicek (misalnya dengan mengukur ukuran kotak per kelas; peringatan 13 kotak sangat kecil juga belum dirinci per kelas). Catatan: toothbrush langka di COCO secara keseluruhan, tetapi di subset ini semua gambar toothbrush ikut terambil, sehingga jumlahnya tidak jauh dari dog.

**Confusion matrix.** Di data val, semua 16 objek dog terdeteksi di E1 maupun E3. Untuk cup, E1 mendeteksi 48 dari 52 dan E3 50 dari 52. Untuk toothbrush (16 objek), E1 mendeteksi 6, salah menjadi cup 4, dan melewatkan 6; E3 mendeteksi 5, tidak ada yang tertukar dengan cup, dan melewatkan 11. Kolom background berisi angka besar (misalnya 2816 tebakan cup di E1 dan 1510 di E3). Kemungkinan besar ini karena confusion matrix dihitung dengan ambang keyakinan sangat rendah pada mode val, sehingga banyak tebakan berkeyakinan rendah ikut terhitung; angka itu tidak berarti model sebanyak itu salah pada pemakaian normal. Meski begitu, E3 punya tebakan palsu lebih sedikit di ketiga kelas (dog 193 ke 151, cup 2816 ke 1510, toothbrush 220 ke 149), sejalan dengan precision yang naik ke 0.869. Angka recall di tabel dan di matriks berbeda karena ambang keyakinannya berbeda.

**Kurva (results.png).**
- Loss train turun hampir terus-menerus di E1 dan E3: cls_loss 3.31 ke 1.06 (E1) dan 2.40 ke 0.84 (E3).
- Loss val naik di epoch-epoch awal (cls_loss val E1 dari 1.93 ke puncak 3.17 di epoch 9; E3 dari 1.46 ke puncak 2.78 di epoch 7) sementara loss train turun, lalu turun lagi dan berakhir di 2.18 (E1) dan 2.28 (E3), masih di atas nilai epoch 1. Pola ini tidak sama dengan kurva ideal di materi; sebagian bisa karena val kecil dan sebagian karena gangguan awal pelatihan seperti dugaan di atas.
- Precision dan recall val bergoyang tajam antar-epoch (precision E1 berayun antara sekitar 0.24 dan 0.88), tanda val 44 gambar tidak stabil.
- mAP val E3 tertinggi setelah epoch 10 ada di epoch 24 (0.239), tidak lama setelah mosaic dimatikan di epoch 21 (close_mosaic=10 pada 30 epoch); apakah itu penyebabnya tidak diuji.

**Hasil di data test.** Model E3 mendapat mAP50 0.630 dan mAP50-95 0.443 di data test (dipakai sekali), sedikit di bawah val (0.657 dan 0.473). Wajar jika test sedikit lebih rendah karena val dipakai untuk memilih model. Namun test hanya 22 gambar dan 37 objek, jadi selisihnya tidak bisa ditafsirkan dengan yakin, dan model yang diuji adalah bobot epoch 1.

**Kecepatan.** Inferensi E1 6.3 ms, E3 9.3 ms, dan E2 3.9 ms. E2 lebih cepat padahal arsitekturnya sama dengan E1; dugaan: model yang belum belajar menghasilkan sedikit kotak sehingga tahap penyaringan kotak lebih ringan. Pengukuran hanya sekali, jadi angka ini kasar.

## 4. Keterbatasan

1. Data kecil (220 gambar, 154 untuk latih), dan val hanya 44 gambar dengan 84 objek (toothbrush hanya 16), sehingga angka per kelas tidak stabil.
2. best.pt E1 dan E3 berasal dari epoch 1, sehingga kurva pelatihan epoch 2-30 tidak menggambarkan model terpilih. Penyebabnya belum diuji.
3. Setiap eksperimen dijalankan sekali dengan satu seed, jadi selisih kecil antar-eksperimen belum bisa dianggap pasti.
4. Data test hanya 22 gambar dan 37 objek.
5. Confusion matrix memakai ambang keyakinan rendah, dan waktu inferensi diukur sekali.
6. Ukuran objek per kelas dan asal 13 kotak sangat kecil belum dianalisis.

## 5. Kesimpulan

Model pretrained COCO (E1) jauh lebih baik daripada model dari nol (E2) pada data sekecil ini: mAP50-95 val 0.341 berbanding 0.001. Menaikkan imgsz ke 640 (E3) memberi tambahan, yaitu mAP50-95 0.473 dan precision 0.869, dengan biaya inferensi sekitar 3 ms lebih lama, sehingga E3 menjadi model terpilih dengan mAP50 0.630 dan mAP50-95 0.443 di data test. Kelas toothbrush paling lemah dan perlu dicek lebih lanjut. Temuan terpenting untuk eksperimen lanjutan: model terbaik muncul di epoch 1 dan tidak terlampaui sampai epoch 30. Dua hal yang masuk akal dicoba berikutnya, belum dijalankan di praktik ini: menurunkan learning rate (dengan optimizer ditetapkan eksplisit supaya lr0 berlaku), dan menambah data kelas yang lemah serta data val.
