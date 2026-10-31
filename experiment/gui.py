"""Tkinter launcher for the supervisor running a session.

Shows a small configuration form pre-filled with the previously used
values (``configs/.last_used.yaml``), so repeat sessions only require
tweaking the participant ID and pressing Apply. On Apply, the window
hides itself and launches the participant-facing pygame screen
(``experiment.main``) as a subprocess; it reappears automatically once
that process exits, and shows the psychometric fit for that session
(from the freshly saved trial CSV) directly in the panel. The session's
exploration heat maps are then generated in the background.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Iterable

import yaml
from PIL import Image, ImageTk

from .config import ExperimentConfig, config_to_dict, load_experiment_config
from .replay_tab import ReplayTab

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
LAST_USED_PATH = CONFIGS_DIR / ".last_used.yaml"

# (attribute, Turkish label, kind)
# kind in {"float", "int", "optional_int", "bool", "mode", "hand", "seconds", "str"}
FIELD_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Katılımcı", [
        ("participant_id", "Katılımcı ID", "str"),
        ("dominant_hand", "Dominant el (keşif eli)", "hand"),
        # Gerçek deney için kapatıldı: oturum her zaman constant_stimuli olarak
        # çalışır (config.py'deki varsayılan), bu yüzden panelde mod seçimi
        # gösterilmiyor. Geri açmak için: aşağıdaki satırın yorumunu kaldırın ve
        # sekme adını yeniden "Katılımcı ve Mod" yapın.
        # ("mode", "Mod", "mode"),
    ]),
    ("Uyaran Geometrisi (mm)", [
        ("base_height_mm", "Referans yükseklik", "float"),
        ("bar_width_mm", "Çubuk genişliği", "float"),
        ("inter_bar_gap_mm", "Çubuklar arası boşluk", "float"),
    ]),
    ("Sabit Uyaranlar Tasarımı", [
        ("delta_max_pct", "Maks. fark oranı (delta_max_pct)", "float"),
        ("n_levels", "Seviye sayısı", "int"),
        ("include_zero_level", "0% seviyeyi dahil et", "bool"),
        ("trials_per_level", "Seviye başına deneme", "int"),
        ("sweeps_per_block", "Blok başına tur (randomizasyon)", "int"),
        ("catch_trial_pct", "Kontrol deneme oranı", "float"),
    ]),
    ("Görev Ayarları", [
        # Zaman sınırı en üstte: oturumdan oturuma en sık ayarlanan alan bu.
        ("response_timeout_s", "Zaman sınırı — yanıt süresi (saniye)", "seconds"),
        ("max_trial_attempts", "Deneme başına maks. gösterim", "int"),
        ("feedback", "Geri bildirim ver", "bool"),
        ("n_practice_trials", "Alıştırma deneme sayısı", "int"),
        ("practice_easiest_levels", "Alıştırmada kullanılacak kolay seviye sayısı", "int"),
        ("practice_pass_fraction", "Alıştırma geçme oranı", "float"),
        ("break_every_n_trials", "Kaç denemede bir mola (0 = mola yok)", "int"),
        ("practice_voice_feedback", "Alıştırmada sesli hız yönlendirmesi", "bool"),
        ("practice_spoken_feedback", "Alıştırmada sesli doğru/yanlış bildirimi", "bool"),
        ("masking_noise", "Kulaklıkta brown noise çal", "bool"),
        ("masking_noise_volume", "Brown noise ses seviyesi (0-1)", "float"),
        ("ideal_finger_speed_mm_s", "İdeal parmak hızı (mm/s)", "float"),
        ("ideal_speed_tolerance_pct", "İdeal hız toleransı (oran)", "float"),
        ("record_main_trace", "Ana deney hareket kaydı", "bool"),
        ("blind_test_mode", "Kör test modu (çubukları gösterme)", "bool"),
    ]),
    ("Donanım / İşleme", [
        ("carrier_freq_hz", "Taşıyıcı frekans (Hz)", "float"),
        ("voltage_peak", "Tepe voltaj (Vpp)", "float"),
        ("ir_sample_hz_nominal", "IR frame örnekleme (Hz)", "float"),
        ("rng_seed", "RNG seed (boş = rastgele)", "optional_int"),
    ]),
    ("Sinyal Zamanlaması", [
        ("signal_watchdog_ms", "Bekçi süresi (ms)", "float"),
    ]),
]

# Yalnızca "mode" alanı geri açılırsa kullanılır (bkz. FIELD_GROUPS).
MODE_CHOICES = ["constant_stimuli"]
HAND_CHOICES = ["", "left", "right"]

# "seconds" alanlarının spinbox sınırları. Panel sınırları; gerçek doğrulama
# config.py'de (response_timeout_s > 0). Üst sınır, yanlışlıkla girilen bir
# 3000'in oturumu kilitlemesini engellemek için var.
SECONDS_MIN = 1.0
SECONDS_MAX = 300.0
SECONDS_STEP = 5.0

# Explanation shown under each field. Where the default traces to something
# concrete (a cited standard/paper, or the rig's actual hardware spec, both
# already documented in experiment/README.md and config.py), that source is
# named directly. Where it doesn't -- most of the task/design numbers below --
# this says so plainly, rather than inventing a citation: those are ordinary
# engineering judgment calls the supervisor is expected to tune per session.
FIELD_HELP: dict[str, str] = {
    "participant_id": "Oturumun katılımcı kodu (ör. P01). Kaydedilen dosya adlarında ve CSV'lerde kullanılır.",
    "dominant_hand": "Katılımcının keşif için kullandığı el (işaret parmağı). Ok tuşları DİĞER elin "
    "altına konur: sağ elini kullanan sola, sol elini kullanan sağa. Kodu etkilemez; config snapshot'ına "
    "ve her CSV satırına yazılır ki analizde el/strateji ayrımı yapılabilsin.",
    "mode": "constant_stimuli: ana sabit-uyaranlar bloğu (tek mod).",
    "base_height_mm": "Referans çubuğun yüksekliği. Karşılaştırma yükseklikleri bunun etrafında "
    "±delta_max_pct ile üretilir; oturum başına sabit tutulan bir tasarım parametresi (bkz. "
    "experiment/README.md: 'One base height ... fixed per session').",
    "bar_width_mm": "Referans ve karşılaştırma çubukları için ortak genişlik. Belirli bir referansa "
    "bağlı değil; Sun ve ark. (2023) elektroadhezyon cihazlarında algılanan minimum çizgi genişliğini "
    "inceliyor, bu değeri seçerken o çalışmaya bakılabilir.",
    "inter_bar_gap_mm": "Çubuklar arası boşluk, >=3.0 mm zorunlu -- bu doğrudan BANA 3.4.3.13 taktil "
    "grafik standardından ve Tang & Beebe (1998)'in çubuklar arası boşluk taban çizgisinden geliyor "
    "(config.py'deki doğrulama kuralına bakın).",
    "delta_max_pct": "En uç karşılaştırma seviyesinin referansa göre yüzde farkı (±); seviyeler bu "
    "değerin negatifinden pozitifine eşit aralıklarla üretilir. Belirli bir makaleye dayanmıyor -- "
    "yaklaşık JND'nin ~1.5 katı civarı önerilir; supervisor her oturumda ayarlar.",
    "n_levels": "delta_max_pct aralığındaki linspace nokta sayısı. TEK sayı verilirse ortadaki 0% "
    "noktası düşer (include_zero_level kapalıyken): 7 ve ±30% → ±10/±20/±30, yani 6 gerçek seviye. "
    "Varsayılan 7 bu yüzden; 6 verilirse seviyeler ±6/±18/±30 olur ve alıştırma ±18'de koşar.",
    "include_zero_level": "0% (referansla birebir aynı) seviyeyi de dahil et. Varsayılan kapalı: "
    "0 farkta 'doğru' cevap tanımsız (şans düzeyinde) olduğundan genelde dışarıda bırakılır.",
    "trials_per_level": "Her seviyede yapılacak deneme sayısı. Varsayılan 10 -- eğri fit'i "
    "(analysis/fit_psychometric.py) varsayılan olarak scipy MLE kullanıyor; bu yöntem az sayıda "
    "denemede daha savrulabilir olduğundan seviye başına en az birkaç, ideal olarak ~10 civarı "
    "deneme önerilir.",
    "sweeps_per_block": "Deneme sırası, her seviyeden eşit sayıda içeren bloklar halinde karıştırılır; "
    "bu alan bir blokta her seviyenin kaç kez geçeceğini belirler. 1 = blok başına her seviyeden 1 "
    "(6 seviyede 6'lık bloklar), 2 = 12'lik bloklar. Tüm seti tek seferde karıştırmak seviyeyi "
    "oturum içindeki zamana bağlı bırakıyordu: bir seviyenin denemeleri ilk/ikinci yarıya ortalama "
    "4.6 farkla, en kötü 10-0 dağılabiliyordu. Bloklama bu farkı tanım gereği sıfırlar. "
    "trials_per_level bu değere tam bölünmeli.",
    "catch_trial_pct": "En uç seviyelerde (±delta_max_pct) eklenen ekstra 'kolay' kontrol denemesi "
    "oranı; katılımcının dikkatini/lapse oranını ölçmek içindir. Standart bir dikkat kontrolü "
    "pratiğidir, belirli bir referansa dayanmıyor.",
    "feedback": "Katılımcıya her denemeden sonra doğru/yanlış bildirilsin mi. Ana blokta genelde "
    "kapalı tutulur ki yanıt yanlılığı oluşmasın; alıştırma denemelerinde ayrıca açık çalıştırılır.",
    "n_practice_trials": "Ana bloktan önce katılımcının göreve alışması için yapılan (geri bildirimli) "
    "deneme sayısı. Alıştırma seviyelerine olabildiğince eşit bölünür; tam bölünmezse artan denemeler "
    "rastgele seçilen seviyelere dağıtılır. 0 = alıştırma bloğu hiç çalışmaz.",
    "practice_easiest_levels": "Alıştırmanın tasarımın EN KOLAY kaç seviyesinden çekileceği. Büyüklük "
    "sayar ve her büyüklük iki işaretiyle birlikte gelir: ±%6/±%18/±%30 bir tasarımda 2 değeri ±%18 ve "
    "±%30 verir, yani 4 işaretli seviye. Alıştırma böylece ana blokta gerçekten karşılaşılacak "
    "seviyeleri kullanır -- eskiden yalnızca en uç seviye kullanılıyordu, katılımcı bir alt seviyeyi "
    "ilk kez test koşullarında görüyordu. Tasarımda olandan fazlası istenirse hepsi kullanılır. "
    "%0 seviyesi asla dahil edilmez: alıştırmada geri bildirim açık ve orada 'doğru' cevap tanımsız.",
    "break_every_n_trials": "Kaç denemede bir mola ekranı gösterileceği. 0 = ana blok kesintisiz "
    "koşar (katılımcı brifingi 'planlı mola yok, konuşma yok' diyor; varsayılan bu yüzden 0).",
    "practice_pass_fraction": "Alıştırma bloğunun geçilmesi için gereken doğru oranı. 0.75 ve 16 "
    "deneme → en az 12 doğru (ceil). Altında kalırsa blok yeniden karılıp baştan koşar; tur sayısı "
    "sınırsız, supervisor ESC ile bitirebilir. Süresi dolan denemeler yanlış sayılır.",
    "practice_spoken_feedback": "Alıştırmada her yanıttan sonra 'Correct' / 'Incorrect' sesli "
    "söylenir. Katılımcı gözü bağlı olduğu için ekrandaki yazı yalnızca operatöre görünür; "
    "sesli bildirim olmadan alıştırmada geri bildirim yok demektir.",
    "masking_noise": "Oturum boyunca kulaklıkta sürekli brown noise çalar (program başlar başlamaz "
    "başlar, çıkışta durur). Cihaz ve klavye seslerini maskeler. Cevap/timeout sesleri ve TTS "
    "bunun üstünden çalar; seviyeyi kulaklıkla test edin.",
    "masking_noise_volume": "Brown noise kanal seviyesi, 0-1 arası. 0.30 ile başlayın; cevap sesi ve "
    "konuşma hâlâ rahat duyulmalı, cihazın tıkırtısı duyulmamalı.",
    "response_timeout_s": "Deneme başına tanınan yanıt süresi; varsayılan 30 saniye. Alıştırma "
    "denemeleri dahil her denemede geçerlidir (katılımcı saati ilk kez ana blokta görmesin diye) "
    "ve ekranda geri sayım olarak gösterilir, son 5 saniyede kırmızıya döner. Süre dolunca yumuşak "
    "bir uyarı sesi çalar ve o deneme sayılmadan bloğun en sonuna ertelenir. Oturumdan oturuma "
    "serbestçe değiştirilebilir; seçilen değer config snapshot'ına da yazılır.",
    "max_trial_attempts": "Bir denemenin katılımcıya en fazla kaç kez gösterileceği (ilk gösterim "
    "dahil). Yanıtsız kalan deneme bloğun sonuna erteleniyor; bu sınır olmasa katılımcı hiç yanıt "
    "vermediğinde oturum hiç bitmezdi. Sınır dolduğunda deneme terk edilir (konsola yazılır, trace "
    "veritabanında timeout olarak kalır) ve o seviyede bir gözlem eksilir. 3 = ilk gösterim + 2 tekrar.",
    "practice_voice_feedback": "Yalnızca alıştırma denemelerinde Faster, Good speed ve Slower "
    "sesli yönlendirmelerini açar. Ana deney bloğunda hiçbir hız sesi çalmaz.",
    "ideal_finger_speed_mm_s": "Alıştırmadaki sesli yönlendirmenin hedef parmak hızı. Varsayılan "
    "100 mm/s, yani 10 cm/s'dir.",
    "ideal_speed_tolerance_pct": "Hızın doğru sayılacağı hedef çevresindeki tolerans oranı. "
    "0.30 değeri, 100 mm/s hedef için 70-130 mm/s aralığı üretir.",
    "record_main_trace": "Alıştırma ve ana deneydeki parmak hareketlerini ayrı bir SQLite trace "
    "dosyasına kaydeder (alıştırma denemeleri is_practice/practice_round ile etiketlenir). "
    "Veritabanı yazımı ayrı worker thread'de yapılır. Animasyon için "
    "analysis/trace_movement.py bu dosyayı okur; oturum bitince analysis/exploration_maps ziyaret ve "
    "oyalanma haritalarını data/exploration_maps altına kendiliğinden kaydeder.",
    "signal_watchdog_ms": "Ana döngüden bu süre boyunca plan gelmezse sinyal kapatılır ve trace'e "
    "signal_watchdog olayı yazılır.",
    "blind_test_mode": "Dokunulacak iki sütun tam ekran yüksekliğinde beyaz şeritlerle işaretlenir "
    "(konumlarını bulmak kör bir arama olmasın diye), ama şeritlerin içindeki gerçek yükseklik "
    "çubukları hiç çizilmez -- yükseklik test edilen şey olduğu için ekrandan okunamamalı, sadece "
    "dokunarak/hissederek karşılaştırılmalı. Dokunma algılama ve sinyal mantığını etkilemez.",
    "carrier_freq_hz": "Elektroadhezyon sinyalinin taşıyıcı frekansı. Bu bir tasarım parametresi değil, "
    "rig'in donanım özelliği -- README'deki donanım tablosunda '125 Hz carrier frequency' olarak "
    "belirtiliyor.",
    "voltage_peak": "Sinyal jeneratörünün tepe voltajı (Vpp, amplifikatör öncesi); README'deki '4V "
    "peak' rig özelliğiyle eşleşir. UYARI: 50x amplifikatörle çarpılınca dokunma yüzeyinde ~100V'a "
    "çıkıyor -- yükseltmeden önce README'nin güvenlik notunu okuyun.",
    "ir_sample_hz_nominal": "IR çerçevenin nominal örnekleme hızı; README'de belirtilen Nexio "
    "NIB170BP donanımının spesifikasyonu (~100 Hz), bir tasarım tercihi değil. Konumun sinyal "
    "kararlarına nasıl girdiği README'nin 'Rendering Method' bölümünde anlatılıyor.",
    "rng_seed": "Deneme sırasını karıştıran rastgele sayı üreteci tohumu. Boş bırakılırsa rastgele "
    "bir tohum seçilir; gerçekleşen tohum yine de tekrarlanabilirlik için config snapshot'ına "
    "kaydedilir.",
}


def _load_initial_config() -> ExperimentConfig:
    if LAST_USED_PATH.exists():
        try:
            return load_experiment_config(LAST_USED_PATH)
        except Exception:
            pass
    return load_experiment_config(DEFAULT_CONFIG_PATH)


class LauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Deney Kontrol Paneli")
        self.root.geometry("760x820")
        self.root.minsize(560, 420)

        # Deliberately left on each platform's native ttk theme (aqua on macOS,
        # vista/winnative on Windows) rather than forcing "clam": on the old
        # Tcl/Tk 8.5 that macOS and some Windows Python installs still ship,
        # forcing a non-native theme has been observed to render ttk widgets
        # fully invisible instead of raising an error.
        style = ttk.Style()
        style.configure("Apply.TButton", font=("TkDefaultFont", 11, "bold"), padding=8)
        style.configure("Header.TLabel", font=("TkDefaultFont", 14, "bold"))
        style.configure("Status.TLabel", foreground="#666666")
        style.configure("Help.TLabel", foreground="#777777", font=("TkDefaultFont", 9))

        self.vars: dict[str, tk.Variable] = {}
        self.process: subprocess.Popen | None = None
        self._plot_image: ImageTk.PhotoImage | None = None
        self._launch_started_at = 0.0
        self._launch_cfg: ExperimentConfig | None = None
        self._launch_dry_run = False
        self._tab_canvases: dict[str, tk.Canvas] = {}
        self._current_canvas: tk.Canvas | None = None
        self._normal_geometry: str | None = None
        self._replay_visible = False

        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        ttk.Label(root, text="Taktil Çubuk Yükseklik Ayrımı Deneyi", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", padx=16, pady=(16, 4)
        )

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 8))

        for group_title, group_fields in FIELD_GROUPS:
            tab_content = self._build_scrollable_tab(self.notebook, group_title)
            for i, (attr, label, kind) in enumerate(group_fields):
                self._build_field(tab_content, i, attr, label, kind)

        self.replay_tab = ReplayTab(self.notebook, DATA_DIR)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        if self.notebook.tabs():
            self._current_canvas = self._tab_canvases.get(self.notebook.tabs()[0])

        self.bottom = ttk.Frame(root, padding=(16, 0, 16, 16))
        self.bottom.grid(row=2, column=0, sticky="ew")

        run_frame = ttk.LabelFrame(self.bottom, text="Çalıştırma Seçenekleri", padding=10)
        run_frame.pack(fill="x", pady=(0, 6))
        self.windowed_var = tk.BooleanVar(value=False)
        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(run_frame, text="Pencereli çalıştır (tam ekran yerine)", variable=self.windowed_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Checkbutton(
            run_frame, text="Dry-run (donanım/pygame yok, sadece uyum kontrolü)", variable=self.dry_run_var
        ).grid(row=1, column=0, sticky="w")

        self.status_var = tk.StringVar(value="")
        ttk.Label(self.bottom, textvariable=self.status_var, style="Status.TLabel").pack(fill="x", pady=(0, 2))

        self.hardware_status_var = tk.StringVar(value="")
        ttk.Label(self.bottom, textvariable=self.hardware_status_var, style="Status.TLabel").pack(
            fill="x", pady=(0, 6)
        )

        console_frame = ttk.LabelFrame(self.bottom, text="Konsol Çıktısı (son oturum)", padding=6)
        console_frame.pack(fill="x", pady=(0, 6))
        self.console_text = tk.Text(
            console_frame, height=6, wrap="word", font=("TkFixedFont", 9), state="disabled"
        )
        console_scroll = ttk.Scrollbar(console_frame, orient="vertical", command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=console_scroll.set)
        self.console_text.pack(side="left", fill="both", expand=True)
        console_scroll.pack(side="right", fill="y")

        plot_frame = ttk.LabelFrame(self.bottom, text="Son Oturum: Psychometric Eğri", padding=10)
        plot_frame.pack(fill="x", pady=(0, 6))
        self.plot_label = ttk.Label(
            plot_frame, text="Henüz bir oturum çalıştırılmadı.", anchor="center", justify="center"
        )
        self.plot_label.pack(fill="both", expand=True)

        button_row = ttk.Frame(self.bottom)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="Varsayılana Sıfırla", command=self._reset_to_default).pack(side="left")
        self.apply_button = ttk.Button(button_row, text="Apply", style="Apply.TButton", command=self._on_apply)
        self.apply_button.pack(side="right")

        self._populate(_load_initial_config())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_scrollable_tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        outer = ttk.Frame(notebook)
        notebook.add(outer, text=title)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content = ttk.Frame(canvas, padding=12)
        content_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _on_content_configure(_event: object) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfig(content_id, width=event.width)

        content.bind("<Configure>", _on_content_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        self._tab_canvases[str(outer)] = canvas
        return content

    def _on_tab_changed(self, event: tk.Event) -> None:
        notebook = event.widget
        selected = notebook.select()
        self._current_canvas = self._tab_canvases.get(selected)
        replay_selected = selected == str(self.replay_tab)
        if replay_selected and not self._replay_visible:
            self._replay_visible = True
            self._normal_geometry = self.root.geometry()
            self.bottom.grid_remove()
            self.root.update_idletasks()
            width = max(1180, self.root.winfo_width())
            height = max(760, self.root.winfo_height())
            self.root.geometry(f"{width}x{height}")
            self.replay_tab.activate()
        elif not replay_selected and self._replay_visible:
            self._replay_visible = False
            self.replay_tab.deactivate()
            self.bottom.grid()
            if self._normal_geometry is not None:
                self.root.geometry(self._normal_geometry)
                self._normal_geometry = None

    def _on_mousewheel(self, event: tk.Event) -> None:
        canvas = self._current_canvas
        if canvas is None:
            return
        delta = event.delta if sys.platform == "darwin" else int(event.delta / 120)
        canvas.yview_scroll(int(-delta), "units")

    def _build_field(self, parent: ttk.Frame, i: int, attr: str, label: str, kind: str) -> None:
        row = i * 2
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=(6, 0))
        if kind == "bool":
            var = tk.BooleanVar()
            ttk.Checkbutton(parent, variable=var).grid(row=row, column=1, sticky="w", pady=(6, 0))
        elif kind == "mode":
            var = tk.StringVar()
            ttk.Combobox(parent, textvariable=var, values=MODE_CHOICES, state="readonly", width=22).grid(
                row=row, column=1, sticky="w", pady=(6, 0)
            )
        elif kind == "hand":
            var = tk.StringVar()
            ttk.Combobox(parent, textvariable=var, values=HAND_CHOICES, state="readonly", width=22).grid(
                row=row, column=1, sticky="w", pady=(6, 0)
            )
        elif kind == "seconds":
            var = tk.StringVar()
            # A spinbox so the limit can be nudged without retyping. ttk.Spinbox
            # needs Tk 8.6; the rig PCs are the reason this file avoids assuming
            # a modern Tk at all, so fall back to a plain entry when it is
            # missing rather than failing to open the panel.
            try:
                widget = ttk.Spinbox(
                    parent,
                    textvariable=var,
                    from_=SECONDS_MIN,
                    to=SECONDS_MAX,
                    increment=SECONDS_STEP,
                    width=22,
                )
            except (AttributeError, tk.TclError):
                widget = ttk.Entry(parent, textvariable=var, width=24)
            widget.grid(row=row, column=1, sticky="w", pady=(6, 0))
        else:
            var = tk.StringVar()
            ttk.Entry(parent, textvariable=var, width=24).grid(row=row, column=1, sticky="w", pady=(6, 0))
        self.vars[attr] = var

        help_text = FIELD_HELP.get(attr)
        if help_text:
            ttk.Label(parent, text=help_text, style="Help.TLabel", wraplength=520, justify="left").grid(
                row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 6)
            )

    def _populate(self, cfg: ExperimentConfig) -> None:
        data = config_to_dict(cfg)
        for attr, var in self.vars.items():
            value = data.get(attr)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set("" if value is None else str(value))

    def _reset_to_default(self) -> None:
        self._populate(load_experiment_config(DEFAULT_CONFIG_PATH))
        self.status_var.set("Varsayılan config yüklendi.")

    def _collect_config(self) -> ExperimentConfig:
        kind_by_attr = {attr: kind for _, group in FIELD_GROUPS for attr, _, kind in group}
        kwargs: dict[str, Any] = {}
        for attr, var in self.vars.items():
            kind = kind_by_attr[attr]
            raw = var.get()
            if kind == "bool":
                kwargs[attr] = bool(raw)
            elif kind in ("str", "mode", "hand"):
                kwargs[attr] = str(raw).strip()
            elif kind == "optional_int":
                text = str(raw).strip()
                kwargs[attr] = int(text) if text else None
            elif kind == "int":
                kwargs[attr] = int(str(raw).strip())
            elif kind == "float":
                kwargs[attr] = float(str(raw).strip())
            elif kind == "seconds":
                seconds = float(str(raw).strip())
                if not SECONDS_MIN <= seconds <= SECONDS_MAX:
                    raise ValueError(
                        f"{attr}: {seconds:g} s, "
                        f"{SECONDS_MIN:g}-{SECONDS_MAX:g} saniye aralığında olmalı"
                    )
                kwargs[attr] = seconds
        return ExperimentConfig(**kwargs)

    def _on_apply(self) -> None:
        self.replay_tab.deactivate()
        try:
            cfg = self._collect_config()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Geçersiz konfigürasyon", str(exc))
            return
        if not cfg.participant_id:
            messagebox.showerror("Eksik alan", "Katılımcı ID boş olamaz.")
            return

        LAST_USED_PATH.parent.mkdir(parents=True, exist_ok=True)
        LAST_USED_PATH.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))

        dry_run = self.dry_run_var.get()
        cmd = [
            sys.executable, "-m", "experiment.main",
            "--config", str(LAST_USED_PATH),
            "--participant", cfg.participant_id,
        ]
        if self.windowed_var.get():
            cmd.append("--windowed")
        if dry_run:
            cmd.append("--dry-run")

        try:
            self.process = subprocess.Popen(cmd, cwd=REPO_ROOT)
        except OSError as exc:
            messagebox.showerror("Başlatılamadı", str(exc))
            return

        self._launch_started_at = time.time()
        self._launch_cfg = cfg
        self._launch_dry_run = dry_run

        self.apply_button.state(["disabled"])
        self.status_var.set(f"Çalışıyor: {cfg.participant_id} ({cfg.mode})...")
        self._plot_image = None
        self.plot_label.configure(image="", text="Oturum çalışıyor...")
        self.root.withdraw()
        threading.Thread(target=self._wait_for_process, daemon=True).start()

    def _on_close(self) -> None:
        self.replay_tab.close()
        self.root.destroy()

    def _wait_for_process(self) -> None:
        assert self.process is not None
        returncode = self.process.wait()
        self.root.after(0, self._on_process_done, returncode)

    def _on_process_done(self, returncode: int) -> None:
        self.process = None
        self.apply_button.state(["!disabled"])
        self.status_var.set(
            "Son oturum tamamlandı." if returncode == 0 else f"Son oturum hata ile bitti (kod {returncode})."
        )
        self.root.deiconify()
        self.root.lift()
        self._show_latest_plot()
        self._start_exploration_maps()

    def _start_exploration_maps(self) -> None:
        """Start the offline heat-map script on the finished session's trace; never raises."""
        cfg = self._launch_cfg
        if cfg is None or self._launch_dry_run or cfg.mode != "constant_stimuli":
            return
        out_dir = DATA_DIR / "exploration_maps"
        try:
            trace = self._newest_since_launch(DATA_DIR.glob(f"{cfg.participant_id}_*_trace.sqlite3"))
            if trace is None:
                return
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / f"{trace.name.removesuffix('_trace.sqlite3')}_maps.log"
            # The experiment process has exited, so the trace is complete; the script only reads it.
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "analysis.exploration_maps", str(trace), "--out", str(out_dir)],
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except Exception as exc:  # noqa: BLE001 -- the session is already saved; only report
            self._set_maps_status(f"Keşif haritaları başlatılamadı: {exc}")
            return
        self._set_maps_status("Keşif haritaları üretiliyor...")
        threading.Thread(target=self._wait_for_maps, args=(process, log_path), daemon=True).start()

    def _wait_for_maps(self, process: subprocess.Popen, log_path: Path) -> None:
        returncode = process.wait()
        if returncode == 0:
            text = f"Keşif haritaları kaydedildi: {log_path.parent}"
        elif returncode == 2:
            text = "Keşif haritası yok: kayıtlı deneme bulunamadı."
        else:
            text = f"Keşif haritaları üretilemedi (kod {returncode}), ayrıntı: {log_path}"
        try:
            self.root.after(0, self._set_maps_status, text)
        except Exception:  # noqa: BLE001 -- the window was closed meanwhile
            pass

    def _set_maps_status(self, text: str) -> None:
        self.status_var.set(f"{self.status_var.get().split(' | ')[0]} | {text}")

    def _newest_since_launch(self, paths: Iterable[Path]) -> Path | None:
        candidates = [p for p in paths if p.stat().st_mtime >= self._launch_started_at - 1]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _show_latest_plot(self) -> None:
        cfg = self._launch_cfg
        if cfg is None:
            return
        try:
            if self._launch_dry_run:
                png_path = self._newest_since_launch(DATA_DIR.glob("dry_run_*.png"))
                if png_path is None:
                    self.plot_label.configure(text="Dry-run grafiği bulunamadı.", image="")
                    return
            else:
                csv_path = self._newest_since_launch(DATA_DIR.glob(f"{cfg.participant_id}_*_trials.csv"))
                if csv_path is None:
                    self.plot_label.configure(
                        text="Bu oturum için deneme verisi bulunamadı (erken çıkılmış olabilir).", image=""
                    )
                    return
                from analysis.fit_psychometric import fit_psychometric, load_session_csvs, plot_psychometric

                levels, n_trials, n_taller = load_session_csvs([csv_path])
                if not levels:
                    self.plot_label.configure(
                        text="Kaydedilen deneme yok (pratik denemeler hariç tutulur).", image=""
                    )
                    return
                fit = fit_psychometric(levels, n_trials, n_taller)
                png_path = csv_path.with_name(csv_path.stem.replace("_trials", "") + "_fit.png")
                plot_psychometric(fit, png_path)
        except Exception as exc:  # noqa: BLE001 -- surfaced in the panel, not fatal to the GUI
            self.plot_label.configure(text=f"Eğri çizilemedi: {exc}", image="")
            return

        image = Image.open(png_path)
        max_width = 620
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, round(image.height * ratio)))
        self._plot_image = ImageTk.PhotoImage(image)
        self.plot_label.configure(image=self._plot_image, text="")


def main() -> None:
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
