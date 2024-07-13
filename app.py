from flask import Flask, request, render_template_string
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model dan vektor TF-IDF yang telah dilatih
model = joblib.load('spam_ham_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# HTML template sebagai string
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Klasifikasi Spam Email</title>
</head>
<body>
    <h1>Klasifikasi Spam Email</h1>
    <form action="/predict" method="post">
        <label for="message">Masukkan pesan Anda:</label><br><br>
        <textarea id="message" name="message" rows="4" cols="50"></textarea><br><br>
        <input type="submit" value="Prediksi">
    </form>
    {% if prediction_text %}
    <h2>{{ prediction_text }}</h2>
    {% endif %}
</body>
</html>
"""

# Mendefinisikan rute untuk merender halaman utama
@app.route('/')
def home():
    return render_template_string(html_template)

# Mendefinisikan rute untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan pesan dari permintaan POST
    message = request.form['message']
    
    # Mentrasnformasi pesan menggunakan vektor TF-IDF
    message_vectorized = vectorizer.transform([message])
    
    # Memprediksi kategori menggunakan model yang telah dilatih
    prediction = model.predict(message_vectorized)
    
    # Memetakan prediksi ke kategori yang sesuai
    category = 'Ham' if prediction[0] == 1 else 'Spam'
    
    # Mengembalikan hasil sebagai respons HTML
    return render_template_string(html_template, prediction_text='Pesan tersebut diklasifikasikan sebagai: {}'.format(category))

if __name__ == '__main__':
    app.run(debug=True)
