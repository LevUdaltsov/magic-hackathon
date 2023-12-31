import os
import re
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO

import PIL.Image
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def send_email_with_image(
    email_address: str, image: PIL.Image, card_image: PIL.Image
) -> str:
    # Remove trailing spaces
    email_address = email_address.strip()

    # Validate email address
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email_address):
        return "Invalid email address"

    # Define email parameters
    subject = "Your Mirror Reading"
    body = "Here is your mirror reading."
    sender_email = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")
    receiver_email = email_address

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    ## Raw image
    # Convert PIL Image to byte stream
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Create MIMEImage and add it to the message
    mime_image = MIMEImage(img_byte_arr, name="mirror_reading.png")
    message.attach(mime_image)

    ## Card image
    # Convert PIL Image to byte stream
    card_img_byte_arr = BytesIO()
    card_image.save(card_img_byte_arr, format="PNG")
    card_img_byte_arr = card_img_byte_arr.getvalue()

    # Create MIMEImage and add it to the message
    mime_card_img = MIMEImage(card_img_byte_arr, name="tarot_card.png")
    message.attach(mime_card_img)

    text = message.as_string()

    # Log in to server using secure context and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

    return "Email sent successfully"
