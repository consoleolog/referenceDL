import smtplib
from email.mime.text import MIMEText
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart


def send_email():
    text = "메일 내용입니다"
    # msg = MIMEText(text)
    msg = MIMEMultipart()
    msg['Subject'] = "이것은 메일제목"
    msg['From'] = 'handsomefergus@naver.com'
    msg['To'] = 'handsomefergus04@gmail.com'
    msg.attach(MIMEText(text, _charset='utf-8'))
    print(msg.as_string())

    with open('./data/travel_data.csv', 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())

    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="data.csv"')
    msg.attach(part)

    s = smtplib.SMTP('smtp.naver.com', 587)
    s.starttls()  # TLS 보안 처리
    s.login('handsomefergus', 'Firstrjtm@@1!')  # 네이버로그인
    s.sendmail('handsomefergus@naver.com', 'handsomefergus04@gmail.com', msg.as_string())
    s.close()

