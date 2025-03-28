from flask import Flask, request, jsonify
from flask_cors import CORS
# from transformers import pipeline
import time

app = Flask(__name__)
CORS(app)  # این خط به فرانت‌اند شما اجازه می‌دهد به API متصل شود

# متغیر جهانی برای ذخیره مدل (فقط یکبار بارگذاری می‌شود)
model = None

def load_model():
    global model
    print("در حال بارگذاری مدل...")
    # model = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
    print("مدل با موفقیت بارگذاری شد!")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running!"})

@app.route('/summarize', methods=['POST'])
def summarize():
    # بررسی اینکه آیا مدل بارگذاری شده است یا خیر
    global model
    if model is None:
        load_model()
    
    # دریافت داده از درخواست
    data = request.json
    
    if not data or 'reviews' not in data:
        return jsonify({"error": "لطفاً نظرات را ارسال کنید"}), 400
    
    reviews = data['reviews']
    min_length = data.get('min_length', 50)
    max_length = data.get('max_length', 150)
    
    # تجمیع همه نظرات در یک متن
    all_reviews = " ".join(reviews)
    
    start_time = time.time()
    
    # خلاصه‌سازی
    summary = model(all_reviews, 
                   min_length=min_length, 
                   max_length=max_length,
                   do_sample=False)[0]['summary_text']
    
    end_time = time.time()
    
    # ساخت پاسخ
    response = {
        "summary": summary,
        "stats": {
            "review_count": len(reviews),
            "total_words": len(all_reviews.split()),
            "summary_words": len(summary.split()),
            "processing_time": round(end_time - start_time, 2)
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    # بارگذاری مدل در زمان شروع
    load_model()
    # اجرای سرور در پورت 5000
    app.run(debug=True, host='0.0.0.0', port=5000)