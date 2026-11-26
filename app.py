import io
import base64
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter

app = Flask(__name__)
# Batasi ukuran upload (5 MB) agar server tidak kebanjiran file besar
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Memuat model CNN yang sudah dilatih (4 kelas)
model = load_model('model_penyakit_cabai.h5', compile=False)

# Urutan kelas sesuai indeks alfabetis pelatihan
KELAS = [
    'Anthracnose',
    'Cercospora Leaf Spot',
    'Fresh Leaf',
    'Leaf Curl Disease'
]

# Informasi gejala dan rekomendasi penanganan real-time
INFO_PENYAKIT = {
    'Anthracnose': {
        'nama_id': 'Antraknosa (Patek)',
        'status': 'Terinfeksi Penyakit',
        'gejala': 'Bercak cokelat kehitaman agak cekung pada daun maupun buah dengan lingkaran konsentris.',
        'penanganan': [
            'Pangkas dan musnahkan bagian daun atau buah yang terinfeksi agar spora jamur tidak menyebar.',
            'Kurangi kelembapan area tanam dengan mengatur jarak tanam dan drainase lahan.',
            'Semprotkan fungisida berbahan aktif mankozeb, propineb, atau azoksistrobin sesuai dosis anjuran.'
        ]
    },
    'Cercospora Leaf Spot': {
        'nama_id': 'Bercak Daun Cercospora (Mata Katak)',
        'status': 'Terinfeksi Penyakit',
        'gejala': 'Bercak bulat kecil berwarna cokelat dengan bagian tengah pucat keabu-abuan menyerupai mata katak.',
        'penanganan': [
            'Petik daun yang berbercak parah lalu jauhkan atau bakar di luar area kebun.',
            'Hindari penyiraman dari atas tajuk daun pada sore hari agar daun tidak lembap semalaman.',
            'Aplikasikan fungisida berbahan aktif klorotalonil, tembaga hidroksida, atau difenokonazol.'
        ]
    },
    'Fresh Leaf': {
        'nama_id': 'Daun Sehat (Normal)',
        'status': 'Sehat',
        'gejala': 'Permukaan daun berwarna hijau segar merata tanpa bercak jamur maupun keriting abnormal.',
        'penanganan': [
            'Pertahankan jadwal penyiraman teratur dan pemupukan berimbang (NPK serta unsur mikro).',
            'Lakukan pemantauan rutin pada bagian bawah daun untuk mencegah serangan hama sejak dini.',
            'Jaga kebersihan gulma di sekitar tanaman cabai.'
        ]
    },
    'Leaf Curl Disease': {
        'nama_id': 'Penyakit Keriting Daun (Virus Gemini / Hama Thrips & Tungau)',
        'status': 'Terinfeksi Penyakit',
        'gejala': 'Helai daun mengeriting, menggulung ke atas atau ke bawah, menebal, dan pertumbuhan pucuk terhambat.',
        'penanganan': [
            'Kendalikan vektor pembawa virus (kutu kebul, thrips, dan tungau) menggunakan insektisida/akarisida berbahan aktif abamektin atau imidakloprid.',
            'Cabut dan musnahkan tanaman yang terinfeksi virus berat sejak awal agar tidak menular ke tanaman sehat.',
            'Pasang perangkap likat kuning (yellow sticky trap) dan berikan pupuk daun untuk memulihkan tunas baru.'
        ]
    }
}

def validasi_daun_cabai(img):
    """
    Validasi warna dan morfologi bentuk daun cabai:
    - Menolak gambar non-tanaman (dominasi warna bukan hijau/kuning daun).
    - Menolak daun menjari/berlekuk dalam seperti daun pepaya atau singkong.
    """
    kecil = img.resize((160, 160))
    arr = np.array(kecil, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 1. Cek komposisi warna daun
    mask_daun = (g > b * 1.05) & (g > r * 0.85) & (g > 35)
    if float(np.mean(mask_daun)) < 0.12:
        return False, 'Gambar tidak terdeteksi sebagai daun tanaman. Unggah foto daun cabai dengan pencahayaan jelas.'

    # Haluskan bintik noise 1-piksel sebelum menghitung panjang tepi
    mask_img = Image.fromarray((mask_daun * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(size=3))
    mask = np.array(mask_img) > 127

    luas = int(np.sum(mask))
    if luas < 400:
        return False, 'Objek daun terlalu kecil atau buram. Ambil foto daun cabai dari jarak lebih dekat.'

    # 2. Cek kompleksitas tepi (keliling / akar luas)
    # Daun cabai utuh berbentuk lonjong (~3.8 - 6.0), sedangkan daun pepaya menjari (> 6.8)
    transisi_h = np.sum(mask[:, 1:] != mask[:, :-1])
    transisi_v = np.sum(mask[1:, :] != mask[:-1, :])
    keliling = float(transisi_h + transisi_v)
    kompleksitas_tepi = keliling / np.sqrt(float(luas))

    if kompleksitas_tepi > 6.8:
        return False, 'Bentuk daun tidak sesuai (terdeteksi daun menjari/berlekuk dalam, bukan daun cabai).'

    return True, None

def prediksi_gambar(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediksi = model.predict(img_array, verbose=0)[0]
    indeks = int(np.argmax(prediksi))
    label = KELAS[indeks]
    akurasi = round(float(prediksi[indeks]) * 100, 2)
    return label, akurasi

def buat_preview(img, ukuran=400):
    """Ubah gambar jadi data URI base64 (hanya di memori, tidak disimpan ke disk)."""
    thumb = img.copy()
    thumb.thumbnail((ukuran, ukuran))
    buf = io.BytesIO()
    thumb.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = None
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            try:
                # Baca file langsung ke memori, TIDAK disimpan ke disk
                img = Image.open(io.BytesIO(file.read())).convert('RGB')
            except Exception:
                error = 'File tidak valid. Unggah gambar berformat JPG, JPEG, atau PNG.'
            else:
                valid, pesan_validasi = validasi_daun_cabai(img)
                if not valid:
                    error = pesan_validasi
                else:
                    label, akurasi = prediksi_gambar(img)
                    if akurasi < 80.0:
                        error = f'Pola daun cabai tidak dikenali secara meyakinkan ({akurasi}%). Pastikan foto yang diunggah adalah daun cabai.'
                    else:
                        detail = INFO_PENYAKIT[label]

                        hasil = {
                            'gambar': buat_preview(img),
                            'kelas': label,
                            'nama_id': detail['nama_id'],
                            'status': detail['status'],
                            'akurasi': akurasi,
                            'gejala': detail['gejala'],
                            'penanganan': detail['penanganan']
                        }

    return render_template('index.html', hasil=hasil, error=error)

@app.errorhandler(413)
def file_terlalu_besar(e):
    return render_template('index.html', hasil=None,
                           error='Ukuran file terlalu besar (maksimal 5 MB).'), 413

app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

@app.route('/panduan')
def panduan():
    return render_template('panduan.html')

@app.route('/penyakit')
def penyakit():
    return render_template('penyakit.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)