from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, db
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model XGBoost yang sudah dilatih
model_path = os.path.join(os.path.dirname(__file__), 'xgboost_model3.pkl')
model = joblib.load(model_path)

# Mengonfigurasi Firebase
firebase_creds = {
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL")
}

cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://himedislogin-default-rtdb.firebaseio.com/'
})

# Mengakses Realtime Database
ref = db.reference('/predictions')  # Sesuaikan dengan path yang sesuai di database

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan data dari request
        data = request.json
        sensor_value_ir = data['sensor_value_ir']
        sensor_value_red = data['sensor_value_red']

        # Mengolah data sensor untuk prediksi
        features = np.array([sensor_value_ir, sensor_value_red]).reshape(1, -1)

        # Melakukan prediksi
        prediction = model.predict(features)[0]

        # Konversi prediksi ke tipe float
        prediction = float(prediction)

        # Membuat data untuk dikirim ke Realtime Database Firebase
        result = {
            'sensor_value_ir': sensor_value_ir,
            'sensor_value_red': sensor_value_red,
            'prediction': prediction
        }

        # Menyimpan data ke Realtime Database
        ref.push(result)

        # Mengirimkan hasil prediksi kembali sebagai JSON
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
