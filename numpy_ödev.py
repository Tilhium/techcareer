import numpy as np
from numpy.lib.stride_tricks import as_strided

print("### TEMEL İŞLEMLER ###\n")

# 1) 0 ile 9 arasındaki tam sayıları içeren dizi
arr1 = np.arange(10)
print(f"1) 0-9 Arası Dizi: {arr1}")

# 2) 3x4 boyutunda sıfır matrisi
arr2 = np.zeros((3, 4))
print(f"\n2) 3x4 Sıfır Matrisi:\n{arr2}")

# 3) 2x3 boyutunda birler matrisi
arr3 = np.ones((2, 3))
print(f"\n3) 2x3 Birler Matrisi:\n{arr3}")

# 4) Dizi özellikleri (shape, size, ndim)
arr4 = np.arange(12).reshape(3, 4)
print(f"\n4) Dizi: \n{arr4}")
print(f"   Boyut (Shape): {arr4.shape}")
print(f"   Eleman Sayısı (Size): {arr4.size}")
print(f"   Eksen Sayısı (ndim): {arr4.ndim}")

# 5) Listeyi int64 NumPy dizisine dönüştürme
liste5 = [10, 20, 30, 40]
arr5 = np.array(liste5, dtype=np.int64)
print(f"\n5) int64 Dizi: {arr5} (Veri Tipi: {arr5.dtype})")

# 6) Koşullu seçim (5 < x < 15)
base_arr6 = np.arange(20)
arr6 = base_arr6[(base_arr6 > 5) & (base_arr6 < 15)]
print(f"\n6) 5'ten büyük 15'ten küçük elemanlar: {arr6}")

# 7) a ve b dizilerini oluşturma
a7 = np.array([1, 2, 3])
b7 = np.array([4, 5, 6])
print(f"\n7) a={a7}, b={b7}")

# 8) Dizi toplamı ve eleman bazında çarpımı
toplam8 = a7 + b7
carpim8 = a7 * b7
print(f"8) Toplam: {toplam8}")
print(f"   Eleman Bazında Çarpım: {carpim8}")

# 9) 5x5 matris oluşturma (1-25 arası)
M9 = np.arange(1, 26).reshape(5, 5)
print(f"\n9) 5x5 Matris (M):\n{M9}")

# 10) Matristen satır ve sütun seçme
satir2 = M9[1, :]  # İkinci satır (indeks 1)
sutun3 = M9[:, 2] # Üçüncü sütun (indeks 2)
print(f"10) İkinci Satır: {satir2}")
print(f"    Üçüncü Sütun: {sutun3}")

# 11-12) Matris Çevirme (Transpose)
arr12 = np.random.randint(1, 10, size=(3, 2))
transpoz12 = arr12.T
print(f"\n12) Orijinal 3x2 Matris:\n{arr12}")
print(f"    Transpozu (2x3):\n{transpoz12}")

# 13-14) Boş Eleman (NaN) Yönetimi
dizi14 = np.array([1, 2, np.nan, 4, 5, np.nan, 7])
nan_sayisi = np.isnan(dizi14).sum()
print(f"\n14) Dizideki NaN Eleman Sayısı: {nan_sayisi}")

# 15) 5x5 rastgele tam sayı matrisi
mat15 = np.random.randint(1, 101, size=(5, 5))
print(f"\n15) 5x5 Rastgele Matris:\n{mat15}")

# 16) İstatistiksel hesaplamalar
ort16 = mat15.mean()
max16 = mat15.max()
std16_axis0 = mat15.std(axis=0)
print(f"16) Ortalama: {ort16:.2f}")
print(f"    En Büyük Değer: {max16}")
print(f"    Sütun Bazlı Std. Sapma: {std16_axis0}")

# 17) Yatay (hstack) ve dikey (vstack) birleştirme
a17 = np.array([1, 2])
b17 = np.array([3, 4])
yatay_birlestirme = np.hstack((a17, b17))
dikey_birlestirme = np.vstack((a17, b17))
print(f"\n17) Yatay Birleştirme (hstack): {yatay_birlestirme}")
print(f"    Dikey Birleştirme (vstack):\n{dikey_birlestirme}")

# 18-19) Matris Çarpımı (Dot Product)
A18 = np.array([[1, 2], [3, 4]])
B18 = np.array([[5, 6], [7, 8]])
matris_carpimi19 = A18 @ B18  # veya np.dot(A18, B18)
print(f"\n19) A Matrisi:\n{A18}")
print(f"    B Matrisi:\n{B18}")
print(f"    Matris Çarpımı (A @ B):\n{matris_carpimi19}")

# 20-21) Veri Merkezleme (Sütun Ortalamasını Çıkarma)
mat20 = np.random.rand(5, 5) * 10
print(f"\n20) Orijinal 5x5 Matris:\n{mat20}")
sutun_ortalamalari21 = mat20.mean(axis=0)
merkezlenmis_mat21 = mat20 - sutun_ortalamalari21
print(f"\n21) Merkezlenmiş Matris (Sütun Ort. ~0):\n{merkezlenmis_mat21}")
print(f"    Yeni Sütun Ortalamaları (Kontrol): {merkezlenmis_mat21.mean(axis=0)}")

# 22-24) Determinant ve Matris Tersi (Inverse)
C22 = np.array([[4, 7], [2, 6]])
print(f"\n22) C Matrisi:\n{C22}")
# 23) Determinant
determinant23 = np.linalg.det(C22)
print(f"23) Determinant: {determinant23}")
# 24) Matris Tersi
ters_matris24 = np.linalg.inv(C22)
print(f"24) Ters Matris (C_inv):\n{ters_matris24}")
birim_matris_kontrolu = C22 @ ters_matris24
print(f"    Kontrol (C @ C_inv) (Birim Matris olmalı):\n{np.round(birim_matris_kontrolu)}")
print(f"    np.allclose ile kontrol: {np.allclose(C22 @ ters_matris24, np.eye(2))}")

# 25) Benzersiz elemanlar ve tekrar sayıları
veri25 = np.array([1, 2, 1, 4, 5, 2, 7, 1])
benzersiz, sayilar = np.unique(veri25, return_counts=True)
print(f"\n25) Veri: {veri25}")
print(f"    Benzersiz Elemanlar: {benzersiz}")
print(f"    Tekrar Sayıları: {sayilar}")

# 26-27) Kırpma (Clipping)
dizi26 = np.random.randn(10) # 100 yerine 10 kullanalım ki görünsün
print(f"\n26) Orijinal Dizi (Normal Dağılım):\n{dizi26}")
kirpilmis_dizi27 = np.clip(dizi26, -2, 2)
print(f"27) -2 ve 2 Aralığına Kırpılmış Dizi:\n{kirpilmis_dizi27}")

# 28-29) İç Çarpım (Inner) ve Dış Çarpım (Outer)
v1_28 = np.array([1, 2, 3])
v2_28 = np.array([4, 5, 6])
print(f"\n28) v1={v1_28}, v2={v2_28}")
# 29)
ic_carpim29 = np.inner(v1_28, v2_28) # Skaler (1*4 + 2*5 + 3*6)
dis_carpim29 = np.outer(v1_28, v2_28) # Matris
print(f"29) İç Çarpım (Nokta Ürün): {ic_carpim29}")
print(f"    Dış Çarpım:\n{dis_carpim29}")

# 30) Büyük Liste ve NumPy Dizisi (Sadece oluşturma)
py_list30 = list(range(1000000))
np_array30 = np.arange(1000000)
print(f"\n30) 1M elemanlı list (py_list30) ve NumPy dizisi (np_array30) oluşturuldu.")

print("\n\n### İLERİ DÜZEY SENARYOLAR ###\n")

# --- Karar Tensöründe Eksen Tabanlı Filtreleme ---
print("--- Karar Tensörü Senaryosu ---")
# (5, 20, 8) boyutunda rastgele bir tensör
tensor31 = np.random.rand(5, 20, 8)
print(f"Tensör boyutu: {tensor31.shape}")

# Her bir zaman adımı (Eksen 1) için ortalama (Eksen 0 ve 2 üzerinden)
# Sonuç (20,) olmalı
zaman_adimi_ort31 = tensor31.mean(axis=(0, 2))
print(f"Her zaman adımının ortalaması (1D Vektör): {zaman_adimi_ort31.shape}")

# Broadcasting: Genel ortalamayı (skaler) her elemandan çıkarma
genel_ort31 = tensor31.mean()
broadcast_sonuc31 = tensor31 - genel_ort31
print(f"Genel ortalama ({genel_ort31:.2f}) çıkarılmış tensör: {broadcast_sonuc31.shape}")

# --- Öklid Uzaklığı (Broadcasting) ---
print("\n--- Öklid Uzaklığı Senaryosu ---")
# A (50, 4) ve B (1, 4)
A32 = np.random.rand(50, 4)
B32 = np.random.rand(1, 4) # Referans noktası

# Broadcasting ile Öklid Uzaklığı
# (A - B)**2 -> (50, 4)
# np.sum(..., axis=1) -> (50,)
# np.sqrt(...) -> (50,)
oklid_uzakliklari32 = np.sqrt(np.sum((A32 - B32)**2, axis=1))
print(f"Tüm noktalara olan uzaklıklar (50 elemanlı): {oklid_uzakliklari32.shape}")

# En kısa mesafenin indeksi
en_kisa_indeks32 = np.argmin(oklid_uzakliklari32)
print(f"Referans noktasına en yakın noktanın indeksi: {en_kisa_indeks32}")

# --- Finansal Volatilite Matrisi ---
print("\n--- Finansal Volatilite Senaryosu ---")
# (120, 6) fiyat matrisi simülasyonu
fiyatlar33 = np.random.rand(120, 6) * 50 + 100 # 100-150 arası fiyatlar

# Günlük getiri (yüzde değişim) matrisi
# (fiyat_bugun - fiyat_dun) / fiyat_dun
# np.diff(axis=0) ile (fiyat_bugun - fiyat_dun) hesaplanır
getiriler33 = np.diff(fiyatlar33, axis=0) / fiyatlar33[:-1, :]
print(f"Getiri matrisi boyutu: {getiriler33.shape}") # (119, 6)

# Kovaryans Matrisi (6x6)
# np.cov, özellikleri satırda bekler (rowvar=True default).
# Bizim verimizde özellikler sütunlarda, bu yüzden rowvar=False
kovaryans33 = np.cov(getiriler33, rowvar=False)
print(f"Kovaryans matrisi boyutu: {kovaryans33.shape}")

# Özdeğerler (Eigenvalues) ve Özvektörler (Eigenvectors)
ozdegerler33, ozvektorler33 = np.linalg.eig(kovaryans33)
print(f"Özdeğerler: {ozdegerler33.shape}")
print(f"Özvektörler: {ozvektorler33.shape}")

# En küçük özdeğere karşılık gelen özvektör
min_ozdeger_indeksi33 = np.argmin(ozdegerler33)
en_az_riskli_vektor33 = ozvektorler33[:, min_ozdeger_indeksi33]
print(f"En az riskli portföyü temsil eden özvektör: {en_az_riskli_vektor33.shape}")

# --- 2D Evrişim Çekirdeği Uygulama (Advanced Slicing) ---
print("\n--- 2D Evrişim Senaryosu ---")
# (20, 20) görüntü ve (3, 3) kernel
goruntu34 = np.random.rand(20, 20)
kernel34 = np.ones((3, 3)) / 9.0

# Döngüsüz evrişim (as_strided kullanarak)
s = goruntu34.strides
view_shape = (18, 18, 3, 3)
view_strides = (s[0], s[1], s[0], s[1])
view34 = as_strided(goruntu34, shape=view_shape, strides=view_strides)
output34 = np.einsum('ijkl,kl->ij', view34, kernel34)


print(f"Evrişim sonucu çıktı boyutu: {output34.shape}")

# --- İklim Anomali Tespiti (3D Eksen Yönetimi) ---
print("\n--- İklim Anomali Senaryosu ---")
# (10 bölge, 30 gün, 2 parametre)
iklim35 = np.random.rand(10, 30, 2)

# Genel ortalamalar (Sıcaklık ve Nem için)
genel_ort35 = iklim35.mean(axis=(0, 1)) # (2,) boyutunda
print(f"Genel ortalamalar (Sıcaklık, Nem): {genel_ort35.shape}")

# Broadcasting ile anomali tespiti
anomali35 = iklim35 - genel_ort35
print(f"Anomali tensörü boyutu: {anomali35.shape}")

# --- Çoklu Eksen Sıralaması (lexsort) ---
print("\n--- Çoklu Eksen Sıralama Senaryosu ---")
# (100, 3) puan matrisi
puanlar36 = np.random.randint(50, 101, size=(100, 3))

# Sıralama: 1. Sınav 3 (indeks 2) azalan, 2. Sınav 1 (indeks 0) artan
keys = (puanlar36[:, 0], -puanlar36[:, 2])
siralama_indeksleri36 = np.lexsort(keys)

sirali_puanlar36 = puanlar36[siralama_indeksleri36]
print("Sıralı Puanlar (Önce Sınav 3 (azalan), sonra Sınav 1 (artan)):")
print(sirali_puanlar36)

# En iyi 10 öğrencinin puanları (sıralı dizinin ilk 10'u)
en_iyi_10_ogrenci36 = sirali_puanlar36[:10]
print(f"\nEn iyi 10 öğrencinin puanları:\n{en_iyi_10_ogrenci36}")

# --- Eşitsiz Gruplama ve İstatistik (Split) ---
print("\n--- Eşitsiz Gruplama Senaryosu ---")
# 1000 elemanlı veri serisi
veri37 = np.random.rand(1000)

# 5 gruba ayırma: [0:50], [50:150], [150:300], [300:500], [500:1000]
split_noktalari = [50, 150, 300, 500]
gruplar37 = np.split(veri37, split_noktalari)

print(f"Grup sayısı: {len(gruplar37)}")
for i, grup in enumerate(gruplar37):
    print(f"  Grup {i+1} boyutu: {len(grup)}")

# Her grubun %90'lık persentil değeri
persentiller37 = [np.percentile(grup, 90) for grup in gruplar37]
print(f"\nHer grubun 90. persentil değerleri:\n{persentiller37}")

# --- Matris Birleştirme ve Eksen Kısıtlaması ---
print("\n--- Matris Birleştirme Senaryosu ---")
# A(5, 10), B(5, 10), C(10, 10)
A38 = np.random.rand(5, 10)
B38 = np.random.rand(5, 10)
C38 = np.random.rand(10, 10)

# A ve B'yi yeni eksende (axis=0) birleştirme
tensor38 = np.stack((A38, B38), axis=0)
print(f"Stack sonucu (A ve B): {tensor38.shape}") # (2, 5, 10)

# Tensörü C ile çarpma (Matris Çarpımı)
sonuc38 = np.einsum('ijk,kl->ijl', tensor38, C38)
print(f"Tensör @ Matris C sonucu (einsum): {sonuc38.shape}")


# --- NaN Değerlerini Koşullu Doldurma ---
print("\n--- NaN Doldurma Senaryosu ---")
# 20x5 matris ve 10 rastgele NaN
veri39 = np.random.rand(20, 5)
nan_indeksleri = np.random.choice(veri39.size, 10, replace=False)
veri39.ravel()[nan_indeksleri] = np.nan
print(f"Oluşturulan NaN sayısı: {np.isnan(veri39).sum()}")

# Sütun medyanları (NaN'ları görmezden gelerek)
medyanlar39 = np.nanmedian(veri39, axis=0)
print(f"Sütun medyanları: {medyanlar39.shape}")

# np.where ile NaN'ları ilgili sütun medyanı ile doldurma

doldurulmus_veri39 = np.where(np.isnan(veri39), medyanlar39, veri39)
print(f"Doldurma sonrası NaN sayısı: {np.isnan(doldurulmus_veri39).sum()}")

# --- Özel Birleştirme ve İndeksleme ---
print("\n--- Özel Birleştirme Senaryosu ---")
# A(10, 3), B(10, 3)
A40 = np.random.rand(10, 3)
B40 = np.random.rand(10, 3)

# Yeni (10, 3) matris: A[:, 0], B[:, 1], A[:, 2]
yeni_matris40 = np.c_[A40[:, 0], B40[:, 1], A40[:, 2]]
print(f"Özel birleştirilmiş matris boyutu: {yeni_matris40.shape}")

# Yeni matris üzerinde istatistikler
sonuc_vektor40 = np.array([
    np.median(yeni_matris40[:, 0]), 
    np.std(yeni_matris40[:, 1]),  
    np.sum(yeni_matris40[:, 2])    
])
print(f"Sonuç vektörü (Medyan, Std, Sum): {sonuc_vektor40}")