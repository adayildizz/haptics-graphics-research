# Kuvvet sensörü: laboratuvar kontrolleri (Aşama 2)

Amaç: Nano17'nin newton cinsinden doğru ölçtüğünü, yükün tamamının sensörden
geçtiğini, sıfır noktasının ne kadar kaydığını ve EA sinyalinin sensörü bozup
bozmadığını görmek. Süre: yaklaşık 45 dakika.

> **Önemli:** NI kartını aynı anda yalnızca **bir** program okuyabilir. Bizim
> komutlarımızı çalıştırmadan önce ATI programını kapat; yoksa "resource is
> reserved" gibi bir hata alırsın.

---

## 0. Hazırlık (5 dk)

1. GitHub Desktop: **Fetch origin**, sonra branch olarak **`force-sensor`**'ı seç.
2. Repo klasöründe bir komut satırı aç ve venv'i etkinleştir.

   PowerShell:
   ```
   cd C:\Users\user\Projects\haptic_experiment_ada
   .\venv\Scripts\Activate.ps1
   ```
   cmd:
   ```
   cd C:\Users\user\Projects\haptic_experiment_ada
   venv\Scripts\activate
   ```
   (venv klasörünün adı farklıysa onu yaz.)
3. `nidaqmx` kurulu mu?
   ```
   python -c "import nidaqmx; print('ok')"
   ```
   `ok` yazmazsa: `pip install nidaqmx`. **NI-DAQmx sürücüsünü güncelleme**
   (PCI-6034E eski bir kart, yeni sürücüler desteği bırakmış olabilir).
4. Kodun çalıştığını kartsız dene:
   ```
   python -m experiment.force_measure --fake --seconds 5 --zero-s 1 --label fake
   ```
5. `.cal` dosyasını bul (ATI programının ayarlarında hangi dosyayı yüklediği
   yazar). Bir kopyasını **repo klasörüne** koy (aşağıdaki komutlar
   `FT05411.cal` adını varsayıyor; adı farklıysa komutlarda değiştir). Bir
   kopyasını da Mac'e getir.
6. Plakanın nasıl durduğunun fotoğrafını çek: sensörden başka destek var mı?

---

## 1. ATI programıyla ağırlık testi (15 dk)

Bizim kodumuzdan bağımsız olarak sensörü ve montajı kontrol eder.

1. Camın üstünde hiçbir şey yokken, ellerin ve kolların plakaya ve çerçeveye
   değmezken **Bias**'a bas. Altı eksen de 0'a yakın olmalı. Ağırlıkların
   altına kâğıt/bez koyacaksan onu Bias'tan **önce** yerleştir.
2. Bir ağırlığı **ortaya** yavaşça koy, 2–3 s bekle, **Fz**'yi oku. Kaldır,
   0'a döndüğünü kontrol et.
3. Aynı ağırlığı sırayla **dört köşeye** ve **iki çubuğun yerine** koy.
4. 2–3 farklı ağırlıkla tekrarla. 200 g'dan ağırına gerek yok (parmak
   kuvvetleri 0,1–1 N arası).

Beklenen Fz = kütle × 9,81:

| Ağırlık | Beklenen Fz |
|---|---|
| 10 g | 0,098 N |
| 20 g | 0,196 N |
| 50 g | 0,49 N |
| 100 g | 0,98 N |
| 200 g | 1,96 N |

Neye bakıyoruz:

- **Her konumda aynı mı?** En önemli soru. Köşelerde belirgin şekilde düşükse
  yükün bir kısmı desteklerden geçiyor.
- **Beklenen değere yakın mı?** Uzaksa eski kalibrasyon ya da montaj sorunlu
  olabilir.
- **Fz'nin işareti:** ağırlık koyunca eksi mi, artı mı? Not et.
- **Fx, Fy** 0'a yakın kalmalı.
- **Tx, Ty** ağırlık köşeye gittikçe büyür: normal (tork = kuvvet × merkezden
  uzaklık).

Doldurulacak tablo:

| Ağırlık | Konum | Fz (okunan) | Fx | Fy | Kaldırınca 0'a döndü mü? |
|---|---|---|---|---|---|
| | orta | | | | |
| | sol üst köşe | | | | |
| | sağ üst köşe | | | | |
| | sol alt köşe | | | | |
| | sağ alt köşe | | | | |
| | sol çubuk | | | | |
| | sağ çubuk | | | | |

---

## 2. Aynı ağırlıklar bizim komutla (5 dk)

ATI programını **kapat**, sonra:

```
python -m experiment.force_measure --cal FT05411.cal --seconds 60 --label weights
```

Önce "eller uzak" der, sonra 60 s canlı yazar. Bu sürede iki ağırlığı ortaya
ve bir köşeye koyup kaldır. Satırın sonundaki parantezde yazan **Fz**
(işaretiyle) ATI programının **Fz**'siyle aynı çıkmalı; **Fn** onun yalnızca
mutlak değeri. Aynıysa kalibrasyonu doğru uyguluyoruz.

---

## 3. Sıfır kayması (5 dk)

```
python -m experiment.force_measure --cal FT05411.cal --seconds 30 --label drift
```

30 s boyunca deneydeki gibi parmağını gezdir ve bastır. Başta ve sonda "eller
uzak" dediğinde elini **tamamen** çek. Sonda sıfırın ne kadar kaydığını newton
olarak yazar. **3 kez** yap. Sonuç, deneyde ne sıklıkla sıfırlayacağımızı
belirleyecek.

---

## 4. EA sinyali ve sürtünme (10 dk)

### 4a. Parmak yokken EA sinyali

```
python -m experiment.force_measure --cal FT05411.cal --seconds 60 --label ea_nofinger
```

Cama dokunma. Jeneratörün çıkışını elle **10 s aç, 10 s kapat**, üç kez.
Canlı değerlere bak: çıkış açılınca Fn ya da Ft değişiyor mu?

- Değişmiyorsa: sorun yok.
- Değişiyorsa: sinyal sensöre parazit yapıyor. Not et.

### 4b. Hocanın istediği ölçüm: EA kapalı/açık sürtünme

```
python -m experiment.force_measure --cal FT05411.cal --seconds 60 --label ea_slide
```

"Eller uzak" adımlarında elini tamamen çek. Kayıt başlayınca işaret
parmağını yüzeyde hafif ve eşit bir baskıyla, sabit hızla ileri geri kaydır;
bu sırada jeneratörün çıkışını **10 s kapalı, 10 s açık**, üç kez. Çıkış
açıkken Ft artmalı; EA etkisinin sayılarla ilk görünüşü.

Çıkışın ne zaman açık olduğu dosyaya kendiliğinden yazılmıyor. Bu yüzden
canlı satırlardaki saniyeye göre değiştir: kayıt **kapalı** başlar,
**10. saniyede aç, 20.'de kapat, 30.'da aç, 40.'ta kapat, 50.'de aç**. Bir
geçişi kaçırırsan hangi saniyede yaptığını not et.

Dosya altı gerinim ölçerin gerilimlerini ham olarak saklar (sıfır noktaları
`_summary.json`'da); teğetsel kuvvetler Fx, Fy, normal kuvvet Fz ve sürtünme
katsayısı μ = √(Fx² + Fy²) / |Fz| sonradan bu dosyadan hesaplanır (Fz'nin
işareti montaja bağlı olduğu için mutlak değeri). μ yalnızca parmağın bastığı
anlar için hesaplanır (|Fz| yaklaşık 0,05 N'luk küçük bir eşiğin üstündeyken),
çünkü Fz sıfıra yakınken μ anlamsız derecede büyür.

---

## 5. Okuyucu açıkken sinyal zamanlayıcısı (2 dk)

```
python -m experiment.signal_scheduler --with-force
```

**PASS** çıkmalı. Son satırları kaydet ya da fotoğrafla.

---

## 6. Getirilecekler

- `experiment/data/` içindeki yeni `force_*` dosyaları (her ölçüm için bir
  `_force.sqlite3` ve bir `_summary.json`)
- `.cal` dosyasının kopyası
- Doldurulmuş ağırlık tablosu
- Plakanın fotoğrafı
- Zamanlayıcı kontrolünün sonucu
