from PIL import Image, ImageDraw, ImageFilter

card_bgs = {
    card_name: Image.open(f"assets/card_{card_name.lower()}.png")
    for card_name in ["death", "magician", "priestess", "sun", "world"]
}

def make_card(img, card_name):
    """
    Make a Tarot card from an image and text.
    """
    card = card_bgs[card_name.lower()]
    card_w = card.size[0]

    img = img.resize((410, 410))

    # vignette image
    margin = 40
    w, h = img.size
    im_a = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(im_a)
    draw.rounded_rectangle(((margin, margin), (w - margin, h - margin)), radius=50, fill=255)
    im_a_blur = im_a.filter(ImageFilter.GaussianBlur(25))

    image = img.copy()
    image.putalpha(im_a_blur)

    # Paste image onto card
    side_margin = card_w // 2 - image.size[0] // 2
    card.paste(image, (side_margin, side_margin - 30), image)

    return card