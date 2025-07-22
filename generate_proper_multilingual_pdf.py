from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def register_fonts():
    pdfmetrics.registerFont(TTFont('NotoSans', 'NotoSans-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('NotoSansCJK', 'NotoSansCJK-Regular.ttc'))

def generate_proper_multilingual_pdf(filename):
    register_fonts()
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setTitle("Multilingual Proper PDF")

    c.setFont("NotoSans", 20)
    c.drawString(50, height - 50, "Introduction to Artificial Intelligence")  # English

    c.setFont("NotoSans", 16)
    c.drawString(50, height - 100, "कृत्रिम बुद्धिमत्ता का परिचय")  # Hindi

    c.drawString(50, height - 150, "人工知能の紹介")  # Japanese

    c.drawString(50, height - 200, "مقدمة في الذكاء الاصطناعي")  # Arabic

    c.drawString(50, height - 250, "செயற்கை நுண்ணறிவின் அறிமுகம்")  # Tamil

    c.drawString(50, height - 300, "人工智能简介")  # Chinese Simplified

    c.setFont("NotoSans", 20)
    c.drawString(50, height - 400, "Deep Learning Overview")  # English H1

    c.showPage()
    c.save()

generate_proper_multilingual_pdf("multilingual_sample_fixed.pdf")
