#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/**
 * @brief Veri tiplerini belirten Enum. 
 * Mikrodenetleyicilerde bellek yönetimi için kritik öneme sahiptir.
 */
typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT16, // Simüle edilmiş (16-bit saklama)
    DTYPE_INT8     // Quantized (8-bit tam sayı)
} DataType;

/**
 * @brief UNION YAPISI: Ödevin en önemli kısmı.
 * Aynı bellek adresini farklı veri tipleri olarak yorumlamamızı sağlar.
 * RAM tasarrufu için 'DataPointer' sadece en büyük tip kadar yer kaplar.
 */
typedef union {
    float *f32;
    uint16_t *f16;
    int8_t *i8;
} DataPointer;

/**
 * @brief Dinamik Tensor Yapısı
 */
typedef struct {
    int *shape;       // Boyut bilgisi (Örn: [2, 3])
    int ndim;         // Boyut sayısı (Rank)
    int size;         // Toplam eleman sayısı (RAM hesaplaması için)
    DataType dtype;   // O anki veri tipi metadata bilgisi
    DataPointer data; // Veriye erişim sağlayan Union
} Tensor;

/**
 * @brief Tensor Oluşturma Fonksiyonu (Dynamic Memory Allocation)
 */
Tensor* create_tensor(int *shape, int ndim, DataType dtype) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    // Boyut bilgisini Heap üzerinde saklıyoruz (Safety)
    t->shape = (int*)malloc(ndim * sizeof(int));
    t->size = 1;
    for(int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    
    t->ndim = ndim;
    t->dtype = dtype;

    // Tip bazlı dinamik bellek ayırma
    switch(dtype) {
        case DTYPE_FLOAT32:
            t->data.f32 = (float*)malloc(t->size * sizeof(float));
            break;
        case DTYPE_FLOAT16:
            t->data.f16 = (uint16_t*)malloc(t->size * sizeof(uint16_t));
            break;
        case DTYPE_INT8:
            t->data.i8 = (int8_t*)malloc(t->size * sizeof(int8_t));
            break;
    }

    return t;
}

/**
 * @brief Quantization İşlemi (Linear Quantization)
 * Float32 değerleri [-128, 127] aralığına sıkıştırır.
 */
void quantize_tensor(Tensor *input, Tensor *output) {
    if (input->dtype != DTYPE_FLOAT32 || output->dtype != DTYPE_INT8) {
        printf("Hata: Tip uyuşmazlığı!\n");
        return;
    }

    // Min-Max tespiti (Dinamik Aralık Belirleme)
    float min_val = input->data.f32[0];
    float max_val = input->data.f32[0];
    for(int i = 1; i < input->size; i++) {
        if(input->data.f32[i] < min_val) min_val = input->data.f32[i];
        if(input->data.f32[i] > max_val) max_val = input->data.f32[i];
    }

    // Scale hesaplama: Veriyi 8-bitlik (-128, 127) uzaya haritalar
    float range = (max_val - min_val);
    float scale = (range == 0) ? 1.0f : range / 255.0f;

    printf("\n--- Debugger Bilgisi: Quantization Parametreleri ---\n");
    printf("Min: %.2f, Max: %.2f, Scale: %.4f\n", min_val, max_val, scale);

    for(int i = 0; i < input->size; i++) {
        float scaled = input->data.f32[i] / scale;
        // Clipping (Sınırlandırma)
        if (scaled > 127) scaled = 127;
        if (scaled < -128) scaled = -128;
        output->data.i8[i] = (int8_t)round(scaled);
    }
}

/**
 * @brief Ekrana Yazdırma Helper Fonksiyonu
 */
void print_tensor(Tensor *t) {
    char *type_name = (t->dtype == DTYPE_FLOAT32) ? "Float32" : "Int8 (Quantized)";
    printf("Tensor (%s) [%d eleman]: ", type_name, t->size);
    for(int i = 0; i < t->size; i++) {
        if(t->dtype == DTYPE_FLOAT32)
            printf("%.2f ", t->data.f32[i]);
        else
            printf("%d ", t->data.i8[i]);
    }
    printf("\n");
}

/**
 * @brief Safe Free: Bellek Sızıntılarını Önleme
 */
void free_tensor(Tensor *t) {
    if (t == NULL) return;
    // Union içindeki hangi pointer'ı free ettiğimiz fark etmez, 
    // hepsi aynı başlangıç adresini gösterir.
    free(t->data.f32); 
    free(t->shape);
    free(t);
}

int main() {
    printf("=== Gömülü Sistemler TinyML Tensor Projesi ===\n");

    // 1. Orijinal Veriyi Hazırla (FP32)
    int shape[] = {1, 6}; 
    Tensor *f32_weights = create_tensor(shape, 2, DTYPE_FLOAT32);
    
    float weights[] = {-0.85f, 0.12f, 0.99f, -1.50f, 0.45f, -0.10f};
    for(int i = 0; i < 6; i++) f32_weights->data.f32[i] = weights[i];

    printf("\n[ADIM 1] Orijinal Ağırlıklar (Yüksek Hassasiyet):\n");
    print_tensor(f32_weights);

    // 2. Quantize Edilmiş Tensor Oluştur (INT8)
    Tensor *i8_weights = create_tensor(shape, 2, DTYPE_INT8);

    // 3. Dönüşüm ve Bellek Tasarrufu
    quantize_tensor(f32_weights, i8_weights);

    printf("\n[ADIM 2] Sıkıştırılmış Veri (Düşük RAM Kullanımı):\n");
    print_tensor(i8_weights);

    // Analiz: Bellek tasarrufunu hesapla
    printf("\n--- Analiz ---\n");
    printf("Float32 Bellek: %lu bytes\n", f32_weights->size * sizeof(float));
    printf("Int8 Bellek: %lu bytes\n", i8_weights->size * sizeof(int8_t));
    printf("Tasarruf Oranı: %%75\n");

    // Temizlik
    free_tensor(f32_weights);
    free_tensor(i8_weights);

    printf("\n✓ Bellek guvenli bir sekilde serbest birakildi.\n");
    return 0;
}