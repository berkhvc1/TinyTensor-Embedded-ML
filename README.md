# TinyTensor: Gömülü Sistemler İçin Bellek Optimize Edilmiş Tensör Kütüphanesi

Bu proje, Arduino ve ESP32 gibi RAM kapasitesi kısıtlı cihazlar üzerinde Yapay Zeka (TinyML) modellerini çalıştırmak için tasarlanmış, bellek dostu bir **Tensör** yapısı uygulamasıdır. 

## 🚀 Özellikler
* **Dinamik Veri Yapısı**: Tek bir bellek alanı üzerinden Float32, Float16 ve Int8 tiplerini destekler.
* **Union Tabanlı Bellek Yönetimi**: Aynı bellek adresini farklı veri tipleri arasında paylaştırarak RAM kullanımını minimize eder.
* **Lineer Quantization (Niceleme)**: Float32 verileri %75 bellek tasarrufu sağlayan Int8 formatına dönüştürür.

## 🛠️ Teknik Mimari

### Union ve Struct Tasarımı
Kütüphanenin kalbi olan `union` yapısı, farklı veri tiplerinin aynı bellek adresini paylaşmasını sağlar.

Quantization (Niceleme) Mantığı
Quantization işlemi, yüksek hassasiyetli float verileri tam sayılara sıkıştırır.

Kullanılan Formül:

P_quantized = round(P_float / Scale)

Bu yöntemle her bir veri noktası bellekte 4 byte yerine sadece 1 byte yer kaplar

Bellek Analizi:

Float32: Her eleman için 4 Byte yer kaplar.

Int8: Her eleman için 1 Byte yer kaplar.

Sonuç: Quantization işlemi ile bellek kullanımında %75 oranında tasarruf sağlanmıştır.

```c
typedef union {
    float *f32;
    uint16_t *f16;
    int8_t *i8;
} DataPointer;
